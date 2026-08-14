#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUSTAGESHAREDMEMORY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// Discardable marker attribute a codegen/front-end attaches to the op whose
// result should be staged through shared memory. It survives
// `convert-triton-to-tritongpu`, so by the time this pass runs the marked
// result already carries a concrete distributed encoding.
constexpr llvm::StringLiteral kStageSharedAttr = "ttg.stage_shared";

// Build a 1-CTA, unswizzled (vec/perPhase/maxPhase = 1) mutable shared-memory
// descriptor for `tensorTy`, mirroring the primitive used elsewhere for
// teenyc-6mv. The physical order is minor-to-major (descending), which is the
// natural default for a contiguous staging buffer.
MemDescType makeSharedMemDescType(RankedTensorType tensorTy) {
  MLIRContext *ctx = tensorTy.getContext();
  int64_t rank = tensorTy.getRank();

  llvm::SmallVector<unsigned> order;
  order.reserve(rank);
  for (int64_t i = rank - 1; i >= 0; --i)
    order.push_back(static_cast<unsigned>(i));

  auto cgaLayout = CGAEncodingAttr::get1CTALayout(ctx, rank);
  auto sharedEnc = SwizzledSharedEncodingAttr::get(
      ctx, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
  auto sharedSpace = SharedMemorySpaceAttr::get(ctx);

  return MemDescType::get(ctx, tensorTy.getShape(), tensorTy.getElementType(),
                          sharedEnc, sharedSpace, /*mutableMemory=*/true,
                          /*allocShape=*/tensorTy.getShape());
}

class TritonGPUStageSharedMemoryPass
    : public impl::TritonGPUStageSharedMemoryBase<
          TritonGPUStageSharedMemoryPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Collect first: we mutate the IR (insert ops, replace uses) inside the
    // loop, so we must not do it while walking.
    llvm::SmallVector<Operation *> marked;
    mod.walk([&](Operation *op) {
      if (op->hasAttr(kStageSharedAttr))
        marked.push_back(op);
    });

    for (Operation *op : marked) {
      op->removeAttr(kStageSharedAttr);

      if (op->getNumResults() != 1) {
        op->emitWarning(kStageSharedAttr)
            << ": expected exactly one result to stage, got "
            << op->getNumResults() << "; skipping";
        continue;
      }

      Value val = op->getResult(0);
      auto tensorTy = dyn_cast<RankedTensorType>(val.getType());
      if (!tensorTy || !tensorTy.getEncoding()) {
        op->emitWarning(kStageSharedAttr)
            << ": result is not a distributed-encoded ranked tensor "
               "(run this pass after convert-triton-to-tritongpu); skipping";
        continue;
      }

      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      Location loc = op->getLoc();

      MemDescType memDescTy = makeSharedMemDescType(tensorTy);
      auto alloc = LocalAllocOp::create(builder, loc, memDescTy);
      auto store = LocalStoreOp::create(builder, loc, val, alloc.getResult());
      auto loaded =
          LocalLoadOp::create(builder, loc, tensorTy, alloc.getResult());

      // Redirect every later consumer of `val` to the reloaded value, but keep
      // the store (which is what writes `val` into shared memory) reading the
      // original.
      llvm::SmallPtrSet<Operation *, 2> keep{store.getOperation(),
                                             alloc.getOperation()};
      val.replaceAllUsesExcept(loaded.getResult(), keep);
    }
  }
};

} // namespace

} // namespace gpu
} // namespace triton
} // namespace mlir
