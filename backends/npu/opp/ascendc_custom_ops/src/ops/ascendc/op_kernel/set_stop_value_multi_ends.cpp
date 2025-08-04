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

class SetStopValueMultiEnds {
public:
    __aicore__ inline SetStopValueMultiEnds(int32_t seqBs, int32_t length)
    {
        // notkens  headDim headNum
        this->seqBs = seqBs;
        this->length = length;
    }

    __aicore__ inline void Process(__gm__ uint8_t *topk_ids, __gm__ uint8_t *stopFlags, __gm__ uint8_t *end_ids,
        __gm__ uint8_t *topk_ids_out, __gm__ uint8_t *stop_flags_out)
    {
        topk_idsGm = (__gm__ int64_t *)topk_ids;
        stopFlagsGm = (__gm__ bool *)stopFlags;
        end_idsGm = (__gm__ int64_t *)end_ids;
        topk_ids_outGm = (__gm__ int64_t *)topk_ids_out;
        stop_flags_outGm = (__gm__ bool *)stop_flags_out;
        for (int i = 0; i < this->seqBs; ++i) {
           *(topk_idsGm + i) = (*(stopFlags + i)) ? *(end_idsGm) : *(topk_idsGm + i);
           *(stop_flags_outGm + i) = *(stopFlagsGm + i);
        }
        pipe_barrier(PIPE_ALL);

        for (int i = 0; i < this->seqBs; ++i) {
            int64_t id = *(topk_idsGm + i);
            bool flag = 0;
            for(int j = 0; j < this->length; ++j) {
                if (id == *(end_idsGm + j)){
                    flag = 1;
                    break;
                }
            }
            if (flag) {
               *(stop_flags_outGm + i) = 1;
            }
        }
        pipe_barrier(PIPE_ALL);
    }

private:
    /* data */
    __gm__ int64_t *topk_idsGm;
    __gm__ bool *stopFlagsGm;
    __gm__ int64_t *end_idsGm;

    __gm__ int64_t *topk_ids_outGm;
    __gm__ bool *stop_flags_outGm;
    
    int32_t seqBs;
    int32_t length;
};
}

extern "C" __global__ __aicore__ void set_stop_value_multi_ends(GM_ADDR topk_ids, GM_ADDR stop_flags, GM_ADDR end_ids,
    GM_ADDR topk_ids_out, GM_ADDR stop_flags_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SetStopValueMultiEnds op(tilingData.seqBs, tilingData.length);
    op.Process(topk_ids, stop_flags, end_ids, topk_ids_out, stop_flags_out);
}
