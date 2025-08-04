#include "kernel_operator.h"
using namespace AscendC;

namespace {

class SetMaskValue {
public:
    __aicore__ inline SetMaskValue(int32_t seqBs, int32_t length)
    {
        this->seqBs = seqBs;
        this->length = length;
    }

    __aicore__ inline void Process(__gm__ uint8_t *inputData, __gm__ uint8_t *stopFlags, __gm__ uint8_t *seqLens,
        __gm__ uint8_t *sequenceLengths)
    {
        inputDataGm = (__gm__ half *)inputData;
        stopFlagsGm = (__gm__ bool *)stopFlags;
        seqLensGm = (__gm__ int32_t *)seqLens;
        sequenceLengthsGm = (__gm__ int32_t *)sequenceLengths;

        for (int32_t i = 0; i < seqBs; i++) {
            if (*(stopFlagsGm + i)) {
                *(sequenceLengthsGm + i) = 0;
                pipe_barrier(PIPE_ALL);
            } else {
                *(sequenceLengthsGm + i) = *(seqLensGm + i);
                pipe_barrier(PIPE_ALL);
            }
            *((inputDataGm + i * length + *(seqLensGm + i))) = (half)1.0;
            pipe_barrier(PIPE_ALL);
        }

    }
private:
    int32_t seqBs;
    int32_t length;
    __gm__ half *inputDataGm;
    __gm__ bool *stopFlagsGm;
    __gm__ int32_t *seqLensGm;
    __gm__ int32_t *sequenceLengthsGm;

};
}

extern "C" __global__ __aicore__ void set_mask_value(GM_ADDR input_data, GM_ADDR stop_flags, GM_ADDR seq_lens,
    GM_ADDR sequence_lengths, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SetMaskValue op(tilingData.seqBs, tilingData.length);
    op.Process(input_data, stop_flags, seq_lens, sequence_lengths);
}
