#include "kernel_operator.h"
using namespace AscendC;

namespace {
// In vector core, repeat size equal to 256B = 128xhalf
const uint32_t REPEATSIZE = 128;
// Default repeat stride is 8
const uint8_t DEFAULREPEATSTRIDE = 8;
const uint32_t BLOCK_SIZE = 16;
// 12288 * 8 means we can process 8 tokens in one loop
const uint64_t MAX_PROCESS_NUM = 12288 * 8;

class SetStopValueMultiEndsV2 {
public:
    __aicore__ inline SetStopValueMultiEndsV2(int32_t bs, int32_t length)
    {
        this->batchNum = bs;
        this->lengthNum = length;
    }

    __aicore__ inline void Init(__gm__ uint8_t *topkIds,
                                   __gm__ uint8_t *stopFlags,
                                   __gm__ uint8_t *seqLens,
                                   __gm__ uint8_t *endIds, 
                                   __gm__ uint8_t *nextTokens,
                                   __gm__ uint8_t *topkIdsOut,
                                   __gm__ uint8_t *stopFlagsOut,
                                   __gm__ uint8_t *nextTokensOut)
    {
        topkIdsGm = (__gm__ int64_t *)topkIds;
        stopFlagsGm = (__gm__ bool *)stopFlags;
        seqLensGm = (__gm__ int32_t *)seqLens;
        endIdsGm = (__gm__ int64_t *)endIds;
        nextTokensGm = (__gm__ int64_t *)nextTokens;
        topkIdsOutGm = (__gm__ int64_t *)topkIdsOut;
        stopFlagsOutGm = (__gm__ bool *)stopFlagsOut;
        nextTokensOutGm = (__gm__ int64_t *)nextTokensOut;
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < batchNum; i++) {
            pipe_barrier(PIPE_ALL);
            if (*(stopFlagsGm + i)) {
                if (*(seqLensGm + i) == 0) {
                    *(topkIdsGm + i) = -1;
                } else {
                    *(topkIdsGm + i) = *endIdsGm;
                    *(nextTokensGm + i) = *endIdsGm;
                }
            } else {
                *(nextTokensGm + i) = *(topkIdsGm + i);
            }

            for (int32_t j = 0; j < lengthNum; j++) {
                if (*(topkIdsGm + i) == *(endIdsGm + j)) {
                    *(stopFlagsGm + i) = true;
                    break;
                }
            }
            pipe_barrier(PIPE_ALL);
        }

        for (int32_t i = 0; i < batchNum; i++) {
            *(topkIdsOutGm + i) = *(topkIdsGm + i);
            *(stopFlagsOutGm + i) = *(stopFlagsGm + i);
            *(nextTokensOutGm + i) = *(nextTokensGm + i);
        }
    }

private:
    __gm__ int64_t *topkIdsGm;
    __gm__ bool *stopFlagsGm;
    __gm__ int32_t *seqLensGm;
    __gm__ int64_t *endIdsGm;
    __gm__ int64_t *nextTokensGm;
    __gm__ int64_t *topkIdsOutGm;
    __gm__ bool *stopFlagsOutGm;
    __gm__ int64_t *nextTokensOutGm;
    
    int32_t batchNum;
    int32_t lengthNum;
};
}

extern "C" __global__ __aicore__ void set_stop_value_multi_ends_v2(GM_ADDR topkIds,
                                                                GM_ADDR stopFlags,
                                                                GM_ADDR seqLens,
                                                                GM_ADDR endIds, 
                                                                GM_ADDR nextTokens,
                                                                GM_ADDR topkIdsOut,
                                                                GM_ADDR stopFlagsOut,
                                                                GM_ADDR nextTokensOut,
                                                                GM_ADDR workspace, 
                                                                GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SetStopValueMultiEndsV2 op(tilingData.bs, tilingData.length);
    op.Init(topkIds, stopFlags, seqLens, endIds, nextTokens, topkIdsOut, stopFlagsOut, nextTokensOut);
    op.Process();
}