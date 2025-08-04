#include "kernel_operator.h"
using namespace AscendC;

class GetMaxLen {
public:
    __aicore__ inline GetMaxLen(int32_t bs)
    {
        this->batchNum = bs;
    }

    __aicore__ inline void Init(__gm__ uint8_t *seqLensEncoder, __gm__ uint8_t *seqLensDecoder, __gm__ uint8_t *seqLensEncoderOut, __gm__ uint8_t *seqLensDecoderOut)
    {
        seqLensEncoderGm = (__gm__ int32_t *)seqLensEncoder;
        seqLensDecoderGm = (__gm__ int32_t *)seqLensDecoder;
        seqLensEncoderOutGm = (__gm__ int32_t *)seqLensEncoderOut;
        seqLensDecoderOutGm = (__gm__ int32_t *)seqLensDecoderOut;
    }

    __aicore__ inline void Process()
    {
        *(seqLensEncoderOutGm) = 0;
        *(seqLensDecoderOutGm) = 0;

        for (int32_t i = 0; i < batchNum; i++) {
            if (*(seqLensEncoderGm + i) > *seqLensEncoderOutGm) {
                *seqLensEncoderOutGm = *(seqLensEncoderGm + i);
            }

            if (*(seqLensDecoderGm + i) > *seqLensDecoderOutGm) {
                *seqLensDecoderOutGm = *(seqLensDecoderGm + i);
            }
        }
        pipe_barrier(PIPE_ALL);
    }

private:
    int32_t batchNum = 0;

    __gm__ int32_t *seqLensEncoderGm;
    __gm__ int32_t *seqLensDecoderGm;
    __gm__ int32_t *seqLensEncoderOutGm;
    __gm__ int32_t *seqLensDecoderOutGm;
};

extern "C" __global__ __aicore__ void get_max_len(GM_ADDR seqLensEncoder, GM_ADDR seqLensDecoder, GM_ADDR seqLensEncoderOut, GM_ADDR seqLensDecoderOut,
                                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GetMaxLen op(tilingData.bs);
    op.Init(seqLensEncoder, seqLensDecoder, seqLensEncoderOut, seqLensDecoderOut);
    op.Process();
}