#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPULOWERINDEXEDSHAREDMEMORY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// Build a 1-CTA, unswizzled (vec/perPhase/maxPhase = 1) mutable shared-memory
// descriptor for a buffer of `shape`/`elemType` with an explicit `order`,
// mirroring StageSharedMemory.cpp's makeSharedMemDescType.
MemDescType makeSharedMemDescType(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                  Type elemType, ArrayRef<unsigned> order) {
  int64_t rank = shape.size();

  auto cgaLayout = CGAEncodingAttr::get1CTALayout(ctx, rank);
  auto sharedEnc = SwizzledSharedEncodingAttr::get(
      ctx, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
  auto sharedSpace = SharedMemorySpaceAttr::get(ctx);

  return MemDescType::get(ctx, shape, elemType, sharedEnc, sharedSpace,
                          /*mutableMemory=*/true, /*allocShape=*/shape);
}

// Physical order minor-to-major (descending) -- the natural default for a
// contiguous, non-transposed staging buffer, e.g. rank 2 -> {1, 0}.
llvm::SmallVector<unsigned> descendingOrder(int64_t rank) {
  llvm::SmallVector<unsigned> order;
  order.reserve(rank);
  for (int64_t i = rank - 1; i >= 0; --i)
    order.push_back(static_cast<unsigned>(i));
  return order;
}

MemDescType makeSharedMemDescType(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                  Type elemType) {
  return makeSharedMemDescType(ctx, shape, elemType, descendingOrder(shape.size()));
}

class TritonGPULowerIndexedSharedMemoryPass
    : public impl::TritonGPULowerIndexedSharedMemoryBase<
          TritonGPULowerIndexedSharedMemoryPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Collect every marker op in program order first: we mutate the IR
    // (insert ops, replace uses, eventually erase) while processing, so we
    // must not do it while walking.
    llvm::SmallVector<Operation *> marked;
    mod.walk([&](Operation *op) {
      if (isa<SharedAllocOp, SharedStoreIndexOp, SharedBarrierOp,
              SharedTransOp, SharedLoadIndexOp>(op))
        marked.push_back(op);
    });

    if (marked.empty())
      return;

    // `original tt.shared_mem SSA value` -> `currently-live ttg.memdesc SSA
    // value` it should be read through. Updated as each shared_alloc /
    // shared_trans is processed; shared_store_index / shared_trans /
    // shared_load_index look up their `buf` operand here.
    llvm::DenseMap<Value, Value> liveMemDesc;
    // `original tt.shared_mem SSA value` -> its static (possibly transposed)
    // shape, needed to build the row/column memdesc type for memdesc_index.
    llvm::DenseMap<Value, llvm::SmallVector<int64_t, 2>> shapeOf;

    llvm::SmallVector<Operation *> toErase;

    for (Operation *markedOp : marked) {
      OpBuilder builder(markedOp);
      Location loc = markedOp->getLoc();

      if (auto alloc = dyn_cast<SharedAllocOp>(markedOp)) {
        MLIRContext *ctx = alloc.getContext();
        ArrayRef<int64_t> shape = alloc.getShape();
        Type elemType = alloc.getElemType();
        MemDescType memDescTy = makeSharedMemDescType(ctx, shape, elemType);

        auto realAlloc = LocalAllocOp::create(builder, loc, memDescTy);
        liveMemDesc[alloc.getResult()] = realAlloc.getResult();
        shapeOf[alloc.getResult()] =
            llvm::SmallVector<int64_t, 2>(shape.begin(), shape.end());
        toErase.push_back(alloc);
        continue;
      }

      if (auto store = dyn_cast<SharedStoreIndexOp>(markedOp)) {
        auto liveIt = liveMemDesc.find(store.getBuf());
        auto shapeIt = shapeOf.find(store.getBuf());
        if (liveIt == liveMemDesc.end() || shapeIt == shapeOf.end()) {
          store.emitError("shared_store_index: buf is not a live indexed "
                          "shared-memory buffer (missing shared_alloc)");
          signalPassFailure();
          return;
        }
        ArrayRef<int64_t> shape = shapeIt->second;
        MemDescType rowTy = makeSharedMemDescType(
            store.getContext(), shape.drop_front(1),
            cast<RankedTensorType>(store.getSrc().getType()).getElementType());

        auto row = MemDescIndexOp::create(builder, loc, rowTy, liveIt->second,
                                          store.getIndex());
        LocalStoreOp::create(builder, loc, store.getSrc(), row.getResult());
        toErase.push_back(store);
        continue;
      }

      if (auto bar = dyn_cast<SharedBarrierOp>(markedOp)) {
        BarrierOp::create(builder, loc, AddrSpace::Local);
        toErase.push_back(bar);
        continue;
      }

      if (auto trans = dyn_cast<SharedTransOp>(markedOp)) {
        auto liveIt = liveMemDesc.find(trans.getBuf());
        auto shapeIt = shapeOf.find(trans.getBuf());
        if (liveIt == liveMemDesc.end() || shapeIt == shapeOf.end()) {
          trans.emitError("shared_trans: buf is not a live indexed "
                          "shared-memory buffer (missing shared_alloc)");
          signalPassFailure();
          return;
        }
        if (shapeIt->second.size() != 2) {
          trans.emitError("shared_trans currently only supports rank-2 buffers");
          signalPassFailure();
          return;
        }
        llvm::SmallVector<int64_t, 2> swapped{shapeIt->second[1],
                                              shapeIt->second[0]};
        Type elemType =
            cast<MemDescType>(liveIt->second.getType()).getElementType();
        // MemDescTransOp infers its result type by permuting the *parent's*
        // order (descendingOrder, e.g. {1, 0}) by the `order` operand
        // ({1, 0}), which for rank 2 yields the reverse -- {0, 1} -- not a
        // fresh descending order for the swapped shape.
        llvm::SmallVector<unsigned> transOrder(
            llvm::reverse(descendingOrder(swapped.size())));
        MemDescType transTy = makeSharedMemDescType(trans.getContext(), swapped,
                                                    elemType, transOrder);

        auto realTrans = MemDescTransOp::create(
            builder, loc, transTy, liveIt->second,
            builder.getDenseI32ArrayAttr({1, 0}));
        liveMemDesc[trans.getResult()] = realTrans.getResult();
        shapeOf[trans.getResult()] = swapped;
        toErase.push_back(trans);
        continue;
      }

      auto load = cast<SharedLoadIndexOp>(markedOp);
      auto liveIt = liveMemDesc.find(load.getBuf());
      auto shapeIt = shapeOf.find(load.getBuf());
      if (liveIt == liveMemDesc.end() || shapeIt == shapeOf.end()) {
        load.emitError("shared_load_index: buf is not a live indexed "
                       "shared-memory buffer (missing shared_alloc)");
        signalPassFailure();
        return;
      }
      ArrayRef<int64_t> shape = shapeIt->second;
      Type elemType =
          cast<RankedTensorType>(load.getResult().getType()).getElementType();
      MemDescType rowTy =
          makeSharedMemDescType(load.getContext(), shape.drop_front(1), elemType);

      auto row = MemDescIndexOp::create(builder, loc, rowTy, liveIt->second,
                                        load.getIndex());
      auto loaded = LocalLoadOp::create(builder, loc, load.getResult().getType(),
                                        row.getResult());
      load.getResult().replaceAllUsesWith(loaded.getResult());
      toErase.push_back(load);
    }

    for (Operation *op : llvm::reverse(toErase))
      op->erase();
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
