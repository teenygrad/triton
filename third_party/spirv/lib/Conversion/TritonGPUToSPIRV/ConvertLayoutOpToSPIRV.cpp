#include "ConvertLayoutOpToSPIRV.h"
#include "Utility.h"

using ::mlir::spirv::delinearize;
using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::spirv::getStridesFromShapeAndOrder;
using ::mlir::spirv::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::DistributedEncodingTrait;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

// Local helper: compute CTA tile shape from a RankedTensorType encoding.
// Replaces the removed free function getShapePerCTATile(Attribute, shape).
static SmallVector<unsigned> getShapePerCTATile(RankedTensorType type) {
  auto llEnc = triton::gpu::toLinearEncoding(type);
  auto spt = llEnc.getSizePerThread();
  auto tpw = llEnc.getThreadsPerWarp();
  auto wpc = llEnc.getWarpsPerCTA();
  SmallVector<unsigned> shape;
  for (auto [s, t, w] : llvm::zip(spt, tpw, wpc))
    shape.push_back(s * t * w);
  return shape;
}

// Local helper: get size per thread for a RankedTensorType.
// Replaces the removed free function getSizePerThread(Attribute).
static SmallVector<unsigned> getSizePerThread(RankedTensorType type) {
  return triton::gpu::toLinearEncoding(type).getSizePerThread();
}

struct ConvertLayoutOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (mlir::isa<DistributedEncodingTrait>(srcLayout) &&
        mlir::isa<DistributedEncodingTrait>(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (isa<NvidiaMmaEncodingAttr>(srcLayout) &&
        isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }
    // Note: shared <-> distributed conversions are now handled via
    // LocalAllocOp/LocalStoreOp/LocalLoadOp, not ConvertLayoutOp.
    // TODO: to be implemented
    llvm_unreachable("unsupported layout conversion");
    return failure();
  }

private:
  SmallVector<Value>
  getMultiDimOffset(Attribute layout, Location loc,
                    ConversionPatternRewriter &rewriter, unsigned elemId,
                    RankedTensorType type,
                    ArrayRef<unsigned> multiDimCTAInRepId,
                    ArrayRef<unsigned> shapePerCTATile) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout)) {
      auto multiDimOffsetFirstElem =
          emitBaseIndexForLayout(loc, rewriter, blockedLayout, type, false);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(type), getOrder(type));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] =
            add(multiDimOffsetFirstElem[d],
                i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                            parentEncoding);
      auto offsets = emitOffsetForLayout(layout, type);
      auto parentOffset = emitOffsetForLayout(parentEncoding, parentTy);
      SmallVector<int> idxs;
      for (SmallVector<unsigned> off : offsets) {
        off.insert(off.begin() + dim, 0);
        auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
        idxs.push_back(std::distance(parentOffset.begin(), it));
      }
      auto multiDimOffsetParent = getMultiDimOffset(
          parentEncoding, loc, rewriter, idxs[elemId], parentTy,
          sliceLayout.paddedShape(multiDimCTAInRepId),
          sliceLayout.paddedShape(shapePerCTATile));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }

    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  SmallVector<Value>
  getWrappedMultiDimOffset(ConversionPatternRewriter &rewriter, Location loc,
                           ArrayRef<Value> multiDimOffset,
                           ArrayRef<unsigned> shape,
                           SmallVector<unsigned> shapePerCTATile,
                           SmallVector<int64_t> shapePerCTA) const {
    unsigned rank = shape.size();
    SmallVector<Value> multiDimOffsetWrapped(rank);
    for (unsigned d = 0; d < rank; ++d) {
      if (shapePerCTATile[d] > shapePerCTA[d])
        multiDimOffsetWrapped[d] = urem(multiDimOffset[d], i32_val(shape[d]));
      else
        multiDimOffsetWrapped[d] = multiDimOffset[d];
    }
    return multiDimOffsetWrapped;
  }

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> origRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(type);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTATiles(rank);
    auto shapePerCTATile = getShapePerCTATile(type);
    auto shapePerCTA = getShapePerCTA(layout, type.getShape());
    auto order = getOrder(type);
    for (unsigned d = 0; d < rank; ++d) {
      numCTATiles[d] = ceil<unsigned>(shapePerCTA[d], shapePerCTATile[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTATiles, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
            rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile,
            shapePerCTA);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
        auto elemPtrTy = ptr_ty(llvmElemTy, spirv::StorageClass::Workgroup);
        Value ptr = gep(elemPtrTy, bitcast(smemBase, elemPtrTy), offset);
        if (vec == 1) {
          if (stNotRd) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread];
            if (isInt1) {
              // spriv::UConvert doesn't support i1
              currVal = select(currVal, int_val(8, 1), int_val(8, 0));
            } else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            store(currVal, ptr);
          } else {
            Value currVal = load(ptr);
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<spirv::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread] = currVal;
          }
        } else {
          auto vecTy = vec_ty(llvmElemTy, vec);
          ptr = bitcast(ptr, ptr_ty(vecTy, spirv::StorageClass::Workgroup));
          if (stNotRd) {
            Value valVec = undef(vecTy);
            for (unsigned v = 0; v < vec; ++v) {
              auto currVal =
                  vals[elemId + linearCTAId * accumSizePerThread + v];
              if (isInt1) {
                // spriv::UConvert doesn't support i1
                currVal = select(currVal, int_val(8, 1), int_val(8, 0));
              } else if (isPtr)
                currVal = ptrtoint(llvmElemTy, currVal);
              valVec = insert_element(vecTy, valVec, currVal, idx_val(v));
            }
            store(valVec, ptr);
          } else {
            Value valVec = load(ptr);
            for (unsigned v = 0; v < vec; ++v) {
              Value currVal = extract_element(llvmElemTy, valVec, idx_val(v));
              if (isInt1)
                currVal = icmp_ne(
                    currVal, rewriter.create<spirv::ConstantOp>(
                                 loc, i8_ty, rewriter.getI8IntegerAttr(0)));
              else if (isPtr)
                currVal = inttoptr(llvmElemTyOrig, currVal);
              vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
            }
          }
        }
      }
    }
  }

  // The MMAV1's result is quite different from the existing "Replica"
  // structure, add a new simple but clear implementation for it to avoid
  // modifying the logic of the existing one.
  void processReplicaForMMAV1(Location loc, ConversionPatternRewriter &rewriter,
                              bool stNotRd, RankedTensorType type,
                              ArrayRef<unsigned> multiDimRepId, unsigned vec,
                              ArrayRef<unsigned> paddedRepShape,
                              ArrayRef<unsigned> outOrd,
                              SmallVector<Value> &vals, Value smemBase,
                              ArrayRef<int64_t> shape,
                              bool isDestMma = false) const {
    assert(0 && "no mma support yet");
  }

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(llvmElemTy, spirv::StorageClass::Workgroup);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    SmallVector<unsigned> origRepShape(rank);
    auto srcShapePerCTATile = getShapePerCTATile(srcTy);
    auto dstShapePerCTATile = getShapePerCTATile(dstTy);
    auto shapePerCTA = getShapePerCTA(srcLayout, shape);

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA =
          std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA =
          std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      origRepShape[d] = maxPerCTA;
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }
    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals = getTypeConverter()->unpackLLElements(loc, adaptor.getSrc(),
                                                     rewriter, srcTy);

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstTy);
    SmallVector<Value> outVals(outElems);

    // Compute inVec and outVec based on contiguous elements per thread
    auto inOrd = getOrder(srcTy);
    unsigned innerDim = rank - 1;
    unsigned inVec = (outOrd[0] != innerDim || inOrd[0] != innerDim)
                         ? 1u
                         : getContigPerThread(srcTy)[inOrd[0]];
    unsigned outVec =
        (outOrd[0] != innerDim) ? 1u : getContigPerThread(dstTy)[outOrd[0]];
    // Clamp to max 128-bit vectorization
    unsigned elemBitWidth = dstTy.getElementTypeBitWidth();
    inVec = std::min(inVec, 128u / elemBitWidth);
    outVec = std::min(outVec, 128u / elemBitWidth);
    // Build paddedRepShape to avoid bank conflicts
    SmallVector<unsigned> paddedRepShape = origRepShape;
    if (rank > 1 &&
        product<unsigned>(origRepShape) != origRepShape[outOrd[0]]) {
      paddedRepShape[outOrd[0]] += std::max(inVec, outVec);
    }

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0)
        barrier();
      if (isa<BlockedEncodingAttr>(srcLayout) ||
          isa<SliceEncodingAttr>(srcLayout) ||
          isa<NvidiaMmaEncodingAttr>(srcLayout)) {
          processReplica(loc, rewriter, /*stNotRd*/ true, srcTy,
                         inNumCTAsEachRep, multiDimRepId, inVec, paddedRepShape,
                         origRepShape, outOrd, vals, smemBase);
      } else {
        assert(0 && "ConvertLayout with input layout not implemented");
        return failure();
      }

      barrier();
      if (isa<BlockedEncodingAttr>(dstLayout) ||
          isa<SliceEncodingAttr>(dstLayout) ||
          isa<NvidiaMmaEncodingAttr>(dstLayout)) {
          processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                         outNumCTAsEachRep, multiDimRepId, outVec,
                         paddedRepShape, origRepShape, outOrd, outVals,
                         smemBase);
      } else {
        assert(0 && "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    Value result =
        getTypeConverter()->packLLElements(loc, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    assert(0 && "no mma support yet");
  }
}; // namespace triton::gpu::ConvertLayoutOp

void populateConvertLayoutOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpSPIRVConversion>(
      typeConverter, context, allocation, indexCacheInfo, benefit);
}
