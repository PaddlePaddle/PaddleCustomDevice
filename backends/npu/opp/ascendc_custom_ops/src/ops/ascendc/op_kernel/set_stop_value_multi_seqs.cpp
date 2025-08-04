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

class SetStopValueMultiSeqs {
public:
    __aicore__ inline SetStopValueMultiSeqs(int32_t bs, int32_t length, int32_t stop_seqs_num, int32_t stop_seqs_max_len, int32_t eos_len)
    {
        this->batchNum = bs;
        this->lengthNum = length;
        this->stopSeqsNum = stop_seqs_num;
        this->stopSeqsMaxLen = stop_seqs_max_len;
        this->eosLen = eos_len;
    }

    __aicore__ inline void Init(__gm__ uint8_t *topkIds,
                                __gm__ uint8_t *preIds,
                                __gm__ uint8_t *stepIdx,
                                __gm__ uint8_t *stopFlags,
                                __gm__ uint8_t *seqLens,
                                __gm__ uint8_t *stopSeqs,
                                __gm__ uint8_t *stopSeqsLen,
                                __gm__ uint8_t *endIds,
                                __gm__ uint8_t *topkIdsOut,
                                __gm__ uint8_t *stopFlagsOut)
    {
        topkIdsGm = (__gm__ int64_t *)topkIds;
        preIdsGm = (__gm__ int64_t *)preIds;
        stepIdxGm = (__gm__ int64_t *)stepIdx;
        stopFlagsGm = (__gm__ bool *)stopFlags;
        seqLensGm = (__gm__ int32_t *)seqLens;
        stopSeqsGm = (__gm__ int64_t *)stopSeqs;
        stopSeqsLenGm = (__gm__ int32_t *)stopSeqsLen;
        endIdsGm = (__gm__ int64_t *)endIds;
        topkIdsOutGm = (__gm__ int64_t *)topkIdsOut;
        stopFlagsOutGm = (__gm__ bool *)stopFlagsOut;
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
                }
            } else {
                for (int32_t j = 0; j < eosLen; j++) {
                    if (*(topkIdsGm + i) == *(endIdsGm + j)) {
                        *(stopFlagsGm + i) = true;
                        *(topkIdsGm + i) = *endIdsGm;
                        break;
                    }
                }
                pipe_barrier(PIPE_ALL);
                if (*(stopFlagsGm + i) == false) {
                    for (int32_t j = 0; j < stopSeqsNum; j++) {
                        if (*(stopSeqsLenGm + j) > 0 && *(stepIdxGm + i) >= *(stopSeqsLenGm + j)) {
                            int32_t sameTokenCount = 0;
                            for (int32_t k = 0; k < *(stopSeqsLenGm + j); k++) {
                                if (k < *(stopSeqsLenGm + j) - 1) {
                                    if (*(preIdsGm + i * lengthNum + *(stepIdxGm + i) - *(stopSeqsLenGm + j) + 1 + k) == *(stopSeqsGm + j * stopSeqsMaxLen + k)) {
                                        sameTokenCount++;
                                    } else {
                                        break;
                                    }
                                } else {
                                    if (*(topkIdsGm + i) == *(stopSeqsGm + j * stopSeqsMaxLen + k)) {
                                        sameTokenCount++;
                                    } else {
                                        break;
                                    }
                                }
                            }
                            if (sameTokenCount == *(stopSeqsLenGm + j)) {
                                *(stopFlagsGm + i) = true;
                                *(topkIdsGm + i) = *endIdsGm;
                                break;
                            }
                        }
                    }
                }
            }
            pipe_barrier(PIPE_ALL);
        }

        for (int32_t i = 0; i < batchNum; i++) {
            *(topkIdsOutGm + i) = *(topkIdsGm + i);
            *(stopFlagsOutGm + i) = *(stopFlagsGm + i);
        }
    }

private:
    __gm__ int64_t *topkIdsGm;
    __gm__ int64_t *preIdsGm;
    __gm__ int64_t *stepIdxGm;
    __gm__ bool *stopFlagsGm;
    __gm__ int32_t *seqLensGm;
    __gm__ int64_t *endIdsGm;
    __gm__ int64_t *stopSeqsGm;
    __gm__ int32_t *stopSeqsLenGm;
    __gm__ int64_t *topkIdsOutGm;
    __gm__ bool *stopFlagsOutGm;
    
    int32_t batchNum;
    int32_t lengthNum;
    int32_t stopSeqsNum;
    int32_t stopSeqsMaxLen;
    int32_t eosLen;
};
}

extern "C" __global__ __aicore__ void set_stop_value_multi_seqs(GM_ADDR topkIds, 
                                                                GM_ADDR preIds,
                                                                GM_ADDR stepIdx,
                                                                GM_ADDR stopFlags,
                                                                GM_ADDR seqLens,
                                                                GM_ADDR stopSeqs,
                                                                GM_ADDR stopSeqsLen,
                                                                GM_ADDR endIds,
                                                                GM_ADDR topkIdsOut,
                                                                GM_ADDR stopFlagsOut,
                                                                GM_ADDR workspace, 
                                                                GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SetStopValueMultiSeqs op(tilingData.bs, tilingData.length, tilingData.stop_seqs_num, tilingData.stop_seqs_max_len, tilingData.eos_len);
    op.Init(topkIds, preIds, stepIdx, stopFlags, seqLens, stopSeqs, stopSeqsLen, endIds, topkIdsOut, stopFlagsOut);
    op.Process();
}