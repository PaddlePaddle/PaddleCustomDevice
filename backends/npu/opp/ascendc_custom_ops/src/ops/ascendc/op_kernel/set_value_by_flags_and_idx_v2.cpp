#include "kernel_operator.h"
using namespace AscendC;

class SetValueByFlagsAndIdxV2 {
public:
    __aicore__ inline SetValueByFlagsAndIdxV2(int32_t bs, int32_t length, int32_t lengthInput)
    {
        this->batchNum = bs;
        this->lengthNum = length;
        this->lengthInputNum = lengthInput;
    }

    __aicore__ inline void Init(__gm__ uint8_t *preIdsAll, 
                                __gm__ uint8_t *inputIds,
                                __gm__ uint8_t *seqLensThisTime,
                                __gm__ uint8_t *seqLensEncoder,
                                __gm__ uint8_t *seqLensDecoder,
                                __gm__ uint8_t *stepIdx,
                                __gm__ uint8_t *stopFlags,
                                __gm__ uint8_t *preIdsAllOut)
    {
        preIdsAllGm = (__gm__ int64_t *)preIdsAll;
        inputIdsGm = (__gm__ int64_t *)inputIds;
        seqLensThisTimeGm = (__gm__ int32_t *)seqLensThisTime;
        seqLensEncoderGm = (__gm__ int32_t *)seqLensEncoder;
        seqLensDecoderGm = (__gm__ int32_t *)seqLensDecoder;
        stepIdxGm = (__gm__ int64_t *)stepIdx;
        stopFlagsGm = (__gm__ bool *)stopFlags;
        preIdsAllOutGm = (__gm__ int64_t *)preIdsAllOut;
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < batchNum; i++) {
            pipe_barrier(PIPE_ALL);
            if (!(*(stopFlagsGm + i))) {
                int32_t seqLenDec = *(seqLensDecoderGm + i);
                int32_t seqLenEnc = *(seqLensEncoderGm + i);
                if ((seqLenDec == 0) && (seqLenEnc == 0)) {
                    continue;
                }
                if (*(stepIdxGm + i) >= 0) {
                    if (seqLenDec == 0) {
                        *(preIdsAllGm + i * lengthNum + *(stepIdxGm + i)) = 
                            *(inputIdsGm + i * lengthInputNum + (seqLenEnc - 1));
                    } else {
                        *(preIdsAllGm + i * lengthNum + *(stepIdxGm + i)) = 
                            *(inputIdsGm + i * lengthInputNum);
                    }
                }
            }
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    __gm__ int64_t *preIdsAllGm;
    __gm__ int64_t *inputIdsGm;
    __gm__ int32_t *seqLensThisTimeGm;
    __gm__ int32_t *seqLensEncoderGm;
    __gm__ int32_t *seqLensDecoderGm;
    __gm__ int64_t *stepIdxGm;
    __gm__ bool *stopFlagsGm;
    __gm__ int64_t *preIdsAllOutGm;

    int32_t batchNum;
    int32_t lengthNum;
    int32_t lengthInputNum;
};

extern "C" __global__ __aicore__ void set_value_by_flags_and_idx_v2(GM_ADDR preIdsAll,
                                                                    GM_ADDR inputIds,
                                                                    GM_ADDR seqLensThisTime,
                                                                    GM_ADDR seqLensEncoder,
                                                                    GM_ADDR seqLensDecoder,
                                                                    GM_ADDR stepIdx,
                                                                    GM_ADDR stopFlags,
                                                                    GM_ADDR preIdsAllOut,
                                                                    GM_ADDR workspace, 
                                                                    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SetValueByFlagsAndIdxV2 op(tilingData.bs, tilingData.length, tilingData.lengthInput);
    op.Init(preIdsAll, inputIds, seqLensThisTime, seqLensEncoder, seqLensDecoder, stepIdx, stopFlags, preIdsAllOut);
    op.Process();
}