#include "kernel_operator.h"
using namespace AscendC;

class SetValueByFlagsAndIdx {
public:
    __aicore__ inline SetValueByFlagsAndIdx(int32_t bs, int32_t length)
    {
        this->lengthNum = length;
        this->batchNum = bs;
    }

    __aicore__ inline void Init(__gm__ uint8_t *preIdsAll, __gm__ uint8_t *preIdsNow, __gm__ uint8_t *stepIdx, __gm__ uint8_t *stopFlags,
                                __gm__ uint8_t *stopFlagsOut)
    {
        preIdsAllGm = (__gm__ int64_t *)preIdsAll;
        preIdsNowGm = (__gm__ int64_t *)preIdsNow;
        stepIdxGm = (__gm__ int64_t *)stepIdx;
        stopFlagsGm = (__gm__ bool *)stopFlags;
        stopFlagsOutGm = (__gm__ bool *)stopFlagsOut;
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < batchNum; i++) {
            *(stopFlagsOutGm + i) = *(stopFlagsGm + i);
            pipe_barrier(PIPE_ALL);

            if (!(*(stopFlagsGm + i))) {
                if (*(stepIdxGm + i) >= 0) {
                    *(preIdsAllGm + i * lengthNum + *(stepIdxGm + i)) = *(preIdsNowGm + i);
                }
            }
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    __gm__ int64_t *preIdsAllGm;
    __gm__ int64_t *preIdsNowGm;
    __gm__ int64_t *stepIdxGm;
    __gm__ bool *stopFlagsGm;
    __gm__ bool *stopFlagsOutGm;

    int32_t batchNum;
    int32_t lengthNum;
};

extern "C" __global__ __aicore__ void set_value_by_flags_and_idx(GM_ADDR preIdsAll, GM_ADDR preIdsNow, GM_ADDR stepIdx, GM_ADDR stopFlags,
                                                                 GM_ADDR stopFlagsOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SetValueByFlagsAndIdx op(tilingData.bs, tilingData.length);
    op.Init(preIdsAll, preIdsNow, stepIdx, stopFlags, stopFlagsOut);
    op.Process();
}
