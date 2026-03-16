#include "TritonGPUToSPIRV.h"
#include "Utility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::spirv::getStridesFromShapeAndOrder;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::MemDescType;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;

struct ReturnOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::ReturnOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::ReturnOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(op, TypeRange(), ValueRange(),
                                                 op->getAttrs());
    return success();
  }
};

struct BroadcastOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::BroadcastOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::BroadcastOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Following the order of indices in the legacy code, a broadcast of:
    //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
    // =>
    //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
    //
    // logically maps to a broadcast within a thread's scope:
    //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
    //   1,spt(k+1)..spt(n-1)]
    // =>
    //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
    //
    // regardless of the order of the layout
    //
    Location loc = op->getLoc();
    Value src = adaptor.getSrc();
    Value result = op.getResult();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(result.getType());
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();

    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcTy);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    SmallVector<Value> srcVals =
        getTypeConverter()->unpackLLElements(loc, src, rewriter, srcTy);

    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.lookup(offset));
    }

    Value resultStruct =
        getTypeConverter()->packLLElements(loc, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct AssertOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::AssertOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::AssertOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto elems = getTypeConverter()->unpackLLElements(
        loc, adaptor.getCondition(), rewriter, op.getCondition().getType());
    auto elemTy = elems[0].getType();
    auto numPerThread = getTotalElemsPerThread(op.getCondition().getType());
    auto shapePerCTA = getShapePerCTA(op.getCondition().getType());
    auto sizePerCTA = 1;
    for (auto s : shapePerCTA)
      sizePerCTA *= s;
    Value condition = int_val(elemTy.getIntOrFloatBitWidth(), 0);
    for (auto &elem : elems) {
      if (elemTy.isSignedInteger() || elemTy.isSignlessInteger()) {
        condition = or_(
            condition,
            logic_cmp_eq(elem, rewriter.create<spirv::ConstantOp>(
                                   loc, elemTy, rewriter.getZeroAttr(elemTy))));
        condition = and_(
            condition, icmp_slt(tid_val(), i32_val(sizePerCTA / numPerThread)));
      } else {
        assert(false && "Unsupported type for assert");
        return failure();
      }
    }
    // Modern AssertOp only has message, no file/func/line
    spirvAssert(op, condition, adaptor.getMessage(), "", "", 0, rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  static void spirvAssert(Operation *op, Value condition, StringRef message,
                          StringRef file, StringRef func, int line,
                          ConversionPatternRewriter &rewriter) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();

    // #block1
    // if (condition) {
    //   #block2
    //   __assertfail(message);
    // }
    // #block3
    Block *prevBlock = op->getBlock();
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto funcOp = getAssertfailDeclaration(rewriter);
    StringRef funcName("__devicelib_assert_fail");
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Value messageString =
        spirv::addStringToModule(loc, rewriter, "assertMessage_", message);
    Value fileString =
        spirv::addStringToModule(loc, rewriter, "assertFile_", file);
    Value funcString =
        spirv::addStringToModule(loc, rewriter, "assertFunc_", func);
    Value lineNumber = i32_val(line);
    Value charSize = int_val(sizeof(size_t) * 8, sizeof(char));
    Value gidVec = getBuiltinVariableValue(
        op, spirv::BuiltIn::GlobalInvocationId, i64_ty, rewriter);
    Value lidVec = getBuiltinVariableValue(
        op, spirv::BuiltIn::LocalInvocationId, i64_ty, rewriter);
    Value gid0 = rewriter.create<spirv::CompositeExtractOp>(
        op->getLoc(), i64_ty, gidVec,
        rewriter.getI32ArrayAttr({static_cast<int32_t>(0)}));
    Value gid1 = rewriter.create<spirv::CompositeExtractOp>(
        op->getLoc(), i64_ty, gidVec,
        rewriter.getI32ArrayAttr({static_cast<int32_t>(1)}));
    Value gid2 = rewriter.create<spirv::CompositeExtractOp>(
        op->getLoc(), i64_ty, gidVec,
        rewriter.getI32ArrayAttr({static_cast<int32_t>(2)}));

    Value lid0 = rewriter.create<spirv::CompositeExtractOp>(
        op->getLoc(), i64_ty, lidVec,
        rewriter.getI32ArrayAttr({static_cast<int32_t>(0)}));
    Value lid1 = rewriter.create<spirv::CompositeExtractOp>(
        op->getLoc(), i64_ty, lidVec,
        rewriter.getI32ArrayAttr({static_cast<int32_t>(1)}));
    Value lid2 = rewriter.create<spirv::CompositeExtractOp>(
        op->getLoc(), i64_ty, lidVec,
        rewriter.getI32ArrayAttr({static_cast<int32_t>(2)}));
    gid0 = bitcast(gid0, ui64_ty);
    gid1 = bitcast(gid1, ui64_ty);
    gid2 = bitcast(gid2, ui64_ty);
    lid0 = bitcast(lid0, ui64_ty);
    lid1 = bitcast(lid1, ui64_ty);
    lid2 = bitcast(lid2, ui64_ty);

    // auto oldInsertPoint = rewriter.saveInsertionPoint();
    // rewriter.setInsertionPointAfter(op);
    auto msgPtr =
        bitcast(messageString, ptr_ty(i8_ty, spirv::StorageClass::Generic));
    auto filePtr =
        bitcast(funcString, ptr_ty(i8_ty, spirv::StorageClass::Generic));
    auto funcPtr =
        bitcast(funcString, ptr_ty(i8_ty, spirv::StorageClass::Generic));
    SmallVector<Value> operands = {msgPtr, filePtr, lineNumber, funcPtr, gid0,
                                   gid1,   gid2,    lid0,       lid1,    lid2};

    // FunctionCallOp void return: use null Type (optional return_value)
    auto ret = rewriter.create<spirv::FunctionCallOp>(
        loc, mlir::Type{}, FlatSymbolRefAttr::get(ctx, funcName), operands);

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, condition, ifBlock, thenBlock);
  }

  static spirv::FuncOp
  getAssertfailDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("__devicelib_assert_fail");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<spirv::FuncOp>(*funcOp);

    // void __devicelib_assert_fail(const char *expr, const char *file,
    //                                         int32_t line, const char *func,
    //                                         uint64_t gid0, uint64_t gid1,
    //                                         uint64_t gid2, uint64_t lid0,
    //                                         uint64_t lid1, uint64_t lid2)
    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType{ptr_ty(i8_ty, spirv::StorageClass::Generic),
                               ptr_ty(i8_ty, spirv::StorageClass::Generic),
                               i32_ty,
                               ptr_ty(i8_ty, spirv::StorageClass::Generic),
                               ui64_ty,
                               ui64_ty,
                               ui64_ty,
                               ui64_ty,
                               ui64_ty,
                               ui64_ty};

    mlir::FunctionType funcType =
        mlir::FunctionType::get(rewriter.getContext(), argsType, {});

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    spirv::FuncOp func = rewriter.create<spirv::FuncOp>(UnknownLoc::get(ctx),
                                                        funcName, funcType);
    auto linkageTypeAttr = rewriter.getAttr<::mlir::spirv::LinkageTypeAttr>(
        spirv::LinkageType::Import);
    auto linkageAttr = rewriter.getAttr<::mlir::spirv::LinkageAttributesAttr>(
        StringAttr::get(rewriter.getContext(), funcName), linkageTypeAttr);
    func.getOperation()->setAttr("linkage_attributes", linkageAttr);

    return func;
  }
};

struct MakeRangeOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::MakeRangeOp> {

  MakeRangeOpSPIRVConversion(
      TritonGPUToSPIRVTypeConverter &converter, MLIRContext *context,
      ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToSPIRVPattern<triton::MakeRangeOp>(
            converter, context, indexCacheInfo, benefit) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto rankedTy = cast<RankedTensorType>(op.getResult().getType());
    auto shape = rankedTy.getShape();
    auto layout = rankedTy.getEncoding();

    auto elemTy = rankedTy.getElementType();
    assert(elemTy.isInteger(32));
    Value start = rewriter.create<spirv::ConstantOp>(
        loc, elemTy, rewriter.getIntegerAttr(elemTy, op.getStart()));
    auto idxs = emitIndices(loc, rewriter, layout, rankedTy);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    Value result =
        getTypeConverter()->packLLElements(loc, retVals, rewriter, rankedTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GetProgramIdOpToSPIRVConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);

    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
        loc, rewriter.getIndexType(), dims[op.getAxisAsInt()]);
    Value blockId_idx =
        rewriter.create<::mlir::arith::TruncIOp>(loc, i32_ty, blockId);
    auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
    auto indexType = typeConverter->getIndexType();

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{i32_ty}, ValueRange{blockId_idx});
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct GetNumProgramsOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::GetNumProgramsOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::GetNumProgramsOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);

    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);

    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct AddPtrOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::AddPtrOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::AddPtrOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto offsetTy = op.getOffset().getType();
    auto ptrTy = op.getPtr().getType();
    auto resultTensorTy = dyn_cast<RankedTensorType>(resultTy);
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType());
      auto ptrs = getTypeConverter()->unpackLLElements(loc, adaptor.getPtr(),
                                                       rewriter, ptrTy);
      auto offsets = getTypeConverter()->unpackLLElements(
          loc, adaptor.getOffset(), rewriter, offsetTy);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(elemTy, ptrs[i], offsets[i]);
      }
      Value view = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
                                                      resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(isa<triton::PointerType>(resultTy));
      Type llResultTy = getTypeConverter()->convertType(resultTy);
      Value result = gep(llResultTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

// Handles ttg.local_alloc: allocates shared memory and returns a MemDesc.
// The MemDesc is represented as a SPIRV struct { ptr, offsets..., strides... }.
struct LocalAllocOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::LocalAllocOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::LocalAllocOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getResult());
    auto memDescTy = cast<MemDescType>(op.getType());
    auto spirvElemTy =
        getTypeConverter()->convertType(memDescTy.getElementType());
    auto elemPtrTy = ptr_ty(spirvElemTy, spirv::StorageClass::Workgroup);
    smemBase = bitcast(smemBase, elemPtrTy);

    auto shapePerCTA = getShapePerCTA(memDescTy);
    auto order = getOrder(memDescTy);

    // Handle 3D tensors: add a leading stride-0 dimension for pipelining
    SmallVector<unsigned> newOrder;
    if (shapePerCTA.size() == 3)
      newOrder = {1 + order[0], 1 + order[1], 0};
    else
      newOrder = SmallVector<unsigned>(order.begin(), order.end());

    auto smemObj =
        SharedMemoryObject(smemBase, shapePerCTA, newOrder, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

// Handles ttg.local_dealloc: no-op in SPIRV (workgroup memory is static).
struct LocalDeallocOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::LocalDeallocOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::LocalDeallocOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

// Handles ttg.local_store: stores a distributed tensor to shared memory.
struct LocalStoreOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::LocalStoreOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::LocalStoreOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<MemDescType>(op.getDst().getType());
    auto srcLayout = srcTy.getEncoding();
    assert(mlir::isa<triton::gpu::BlockedEncodingAttr>(srcLayout) &&
           "LocalStoreOp expects blocked src layout");

    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getDst(), rewriter);
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto shapePerCTA = getShapePerCTA(dstTy);

    auto inOrd = getOrder(srcTy);
    auto outOrd = getOrder(dstTy);
    auto dstStrides =
        getStridesFromShapeAndOrder(shapePerCTA, outOrd, loc, rewriter);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy, false);
    storeDistributedToShared(op.getSrc(), adaptor.getSrc(), dstStrides,
                             srcIndices, op.getDst(), smemObj.base, elemTy,
                             loc, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

// Handles ttg.local_load: loads from shared memory into a distributed tensor.
struct LocalLoadOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::LocalLoadOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::LocalLoadOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto dstLayout = dstTy.getEncoding();

    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), rewriter);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());

    auto inOrd = getOrder(srcTy);
    auto srcStrides =
        getStridesFromShapeAndOrder(getShapePerCTA(srcTy), inOrd, loc, rewriter);
    auto dstIndices = emitIndices(loc, rewriter, dstLayout, dstTy);

    SmallVector<Value> outVals =
        loadSharedToDistributed(op.getResult(), dstIndices, op.getSrc(),
                                smemObj, elemTy, loc, rewriter);
    Value result =
        getTypeConverter()->packLLElements(loc, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct AsyncWaitOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::AsyncWaitOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::AsyncWaitOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: implement the async memory fetch.
#if 0
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);
#endif
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCommitGroupOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: implement the async memory fetch.
#if 0
    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));
    // Safe to remove the op since it doesn't have any return value.
#endif
    rewriter.eraseOp(op);
    return success();
  }
};

void populateTritonGPUToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<AddPtrOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<AsyncCommitGroupOpSPIRVConversion>(typeConverter, context,
                                                  benefit);
  patterns.add<AsyncWaitOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<BroadcastOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<GetProgramIdOpToSPIRVConversion>(typeConverter, context,
                                                benefit);
  patterns.add<GetNumProgramsOpSPIRVConversion>(typeConverter, context,
                                                benefit);
  patterns.add<LocalAllocOpSPIRVConversion>(typeConverter, context, allocation,
                                            benefit);
  patterns.add<LocalDeallocOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<LocalLoadOpSPIRVConversion>(typeConverter, context, allocation,
                                           benefit);
  patterns.add<LocalStoreOpSPIRVConversion>(typeConverter, context, allocation,
                                            benefit);
  patterns.add<MakeRangeOpSPIRVConversion>(typeConverter, context,
                                           indexCacheInfo, benefit);
  patterns.add<ReturnOpSPIRVConversion>(typeConverter, context, benefit);
  //  patterns.add<PrintfOpConversion>(typeConverter, benefit);
  patterns.add<AssertOpSPIRVConversion>(typeConverter, context, benefit);
}
