#include "kernel_operator.h"
using namespace AscendC;

const int32_t BUFFER_NUM = 1;

class TokenPenaltyMultiScoresWithStopSeqs {
public:
    __aicore__ inline TokenPenaltyMultiScoresWithStopSeqs(
        int32_t vs, int32_t vsBlock, int32_t sl, int32_t stop_seqs_num, int32_t stop_seqs_max_len, int32_t eos_len, int32_t bs, int32_t bsBlock)
    {
        vocabSize_ = vs;
        vocabLocalSize_ = vsBlock;
        seqLen_ = sl;
        stopSeqsNum = stop_seqs_num;
        stopSeqsMaxLen = stop_seqs_max_len;
        eosLen = eos_len;
        bs_ = bs;
        bsEachCore_ = bsBlock;
    }

    __aicore__ inline void Init(__gm__ uint8_t *preIds,
                                __gm__ uint8_t *logitsIn,
                                __gm__ uint8_t *repeatTimes,
                                __gm__ uint8_t *penaltyScores,
                                __gm__ uint8_t *frequencyScores,
                                __gm__ uint8_t *presenceScores,
                                __gm__ uint8_t *curLen,
                                __gm__ uint8_t *minLen,
                                __gm__ uint8_t *stopSeqs,
                                __gm__ uint8_t *stopSeqsLen,
                                __gm__ uint8_t *eosTokenId,
                                __gm__ uint8_t *logitsOut)
    {
        preIdsGm_ = (__gm__ int64_t *)preIds;
        logitsInGm_ = (__gm__ float *)logitsIn;
        repeatTimesGm_ = (__gm__ int32_t *)repeatTimes;
        penaltyScoresGm_ = (__gm__ float *)penaltyScores;
        frequencyScoresGm_ = (__gm__ float *)frequencyScores;
        presenceScoresGm_ = (__gm__ float *)presenceScores;
        curLenGm_ = (__gm__ int64_t *)curLen;
        minLenGm_ = (__gm__ int64_t *)minLen;
        stopSeqsGm_ = (__gm__ int64_t *)stopSeqs;
        stopSeqsLenGm_ = (__gm__ int32_t *)stopSeqsLen;
        eosTokenIdGm_ = (__gm__ int64_t *)eosTokenId;
        logitsOutGm_ = (__gm__ float *)logitsOut;

        pipe_.InitBuffer(logitsInQueue_, BUFFER_NUM, vocabLocalSize_ * sizeof(float));
        pipe_.InitBuffer(repeatTimesInQueue_, BUFFER_NUM, vocabLocalSize_ * sizeof(int32_t));
        pipe_.InitBuffer(logitsOutQueue_, BUFFER_NUM, vocabLocalSize_ * sizeof(float));
        pipe_.InitBuffer(repeatTimesFp32Buff_, vocabLocalSize_ * sizeof(float));
        pipe_.InitBuffer(repeatTimesUint8Buff_, vocabLocalSize_ * sizeof(uint8_t));
        pipe_.InitBuffer(repeatTimesTmpBuff_, vocabLocalSize_ * sizeof(float));
        pipe_.InitBuffer(logitsTmpBuff_, vocabLocalSize_ * sizeof(float));
        pipe_.InitBuffer(logitsTmpBuff2_, vocabLocalSize_ * sizeof(float));
    }
/**
    preIds: bs, seqLen
    logits: bs, vocabSize
    repeatTimes: bs, vocabSize
    penaltyScores: bs
    frequencyScores: bs
    presenceScores: bs
    curLen: bs
    minLen: bs
    eosTokenId: end_length, 
    logitsOut: bs, vocabSize
*/
    __aicore__ inline void Process()
    {
        int32_t vocLoop = (vocabSize_ + vocabLocalSize_ - 1) / vocabLocalSize_;
        for (int32_t bsIdEachCore = 0; bsIdEachCore < bsEachCore_; bsIdEachCore++) {
            for (int32_t i = 0; i < vocLoop; i++) {
                int32_t bsOffset = GetBlockIdx() * bsEachCore_ + bsIdEachCore;
                if (bsOffset >= bs_) {
                    break;
                }
                int32_t gmOffset = bsOffset * vocabSize_ + i * vocabLocalSize_;
                logitsInG_.SetGlobalBuffer(logitsInGm_ + gmOffset);
                logitsOutG_.SetGlobalBuffer(logitsOutGm_ + gmOffset);
                repeateTimesG_.SetGlobalBuffer(repeatTimesGm_ + gmOffset);
                CopyIn();
                Compute(bsOffset, i, vocabLocalSize_);
                CopyOut();
            }
        }
    }

    __aicore__ inline void CopyIn()
    {
        LocalTensor<float> logitsInL = logitsInQueue_.AllocTensor<float>();
        LocalTensor<int32_t> repeatTimesL = repeatTimesInQueue_.AllocTensor<int32_t>();
        DataCopy(logitsInL, logitsInG_, vocabLocalSize_);
        DataCopy(repeatTimesL, repeateTimesG_, vocabLocalSize_);
        logitsInQueue_.EnQue(logitsInL);
        repeatTimesInQueue_.EnQue(repeatTimesL);
    }

    __aicore__ inline void Compute(int32_t bsId, int32_t vocSizeId, int32_t vocabLocalSize_)
    {
        LocalTensor<float> logitsInL = logitsInQueue_.DeQue<float>();
        LocalTensor<int32_t> repeatTimesL = repeatTimesInQueue_.DeQue<int32_t>();

        int64_t startVocId = vocabLocalSize_ * vocSizeId;
        if (*(curLenGm_ + bsId) >= 0) {
            // min_length_logits_process
            if (*(curLenGm_ + bsId) < *(minLenGm_ + bsId)) {
                for (int32_t i = 0; i < stopSeqsNum; i++) {
                    for (int32_t j = 0; j < *(stopSeqsLenGm_ + i); j++) {
                        int64_t eosTokenIdOffset = *(stopSeqsGm_ + i * stopSeqsMaxLen + j) - startVocId;
                        pipe_barrier(PIPE_ALL);
                        if (eosTokenIdOffset >= 0 && eosTokenIdOffset < vocabLocalSize_) {
                            logitsInL.SetValue(eosTokenIdOffset, (float)-1e10);
                            pipe_barrier(PIPE_ALL);
                        }
                    }
                }
                for (int i = 0; i < eosLen; ++i) {
                    int64_t eosTokenIdOffset = *(eosTokenIdGm_ + i) - startVocId;
                    pipe_barrier(PIPE_ALL);
                    if (eosTokenIdOffset >= 0 && eosTokenIdOffset < vocabLocalSize_) {
                        logitsInL.SetValue(eosTokenIdOffset, (float)-1e10);
                        pipe_barrier(PIPE_ALL);
                    }
                }
            }

            // update_repeat_times
            for (int i = 0; i < seqLen_; i++) {
                int64_t predId = *(preIdsGm_ + bsId * seqLen_ + i);
                if (predId < 0) {
                    break;
                }
                int64_t predIdOffset = predId - startVocId;
                pipe_barrier(PIPE_ALL);
                if (predIdOffset >= 0 && predIdOffset < vocabLocalSize_) {
                    int32_t repeatNew = repeatTimesL.GetValue(predIdOffset) + 1;
                    repeatTimesL.SetValue(predIdOffset, repeatNew);
                    pipe_barrier(PIPE_ALL);
                }
            }
        }

        LocalTensor<float> logitsOutL = logitsOutQueue_.AllocTensor<float>();
        LocalTensor<float> logitsInTmpL = logitsTmpBuff_.Get<float>();
        LocalTensor<float> logitsInTmpL2 = logitsTmpBuff2_.Get<float>();
        LocalTensor<float> repeatTimesFp32L = repeatTimesFp32Buff_.Get<float>();
        LocalTensor<uint8_t> repeatTimesUint8L = repeatTimesUint8Buff_.Get<uint8_t>();
        LocalTensor<float> repeatTimesTmpL = repeatTimesTmpBuff_.Get<float>();

        // update_value_by_repeat_times
        float alpha = *(penaltyScoresGm_ + bsId);
        float alphaR = 1.0f / alpha;
        float beta = *(frequencyScoresGm_ + bsId);
        float gamma = *(presenceScoresGm_ + bsId);

        Cast(repeatTimesFp32L, repeatTimesL, RoundMode::CAST_NONE, vocabLocalSize_);
        Duplicate(repeatTimesTmpL, 0.5f, vocabLocalSize_);
        Compare(repeatTimesUint8L, repeatTimesFp32L, repeatTimesTmpL, CMPMODE::GT, vocabLocalSize_);
        Muls(repeatTimesTmpL, repeatTimesFp32L, beta, vocabLocalSize_);
        Adds(repeatTimesTmpL, repeatTimesTmpL, gamma, vocabLocalSize_);
        Select(repeatTimesFp32L, repeatTimesUint8L, repeatTimesTmpL, 0.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, vocabLocalSize_);
        pipe_barrier(PIPE_ALL);

        Maxs(logitsInTmpL, logitsInL, 0.0f, vocabLocalSize_);
        Duplicate(logitsInTmpL2, alphaR, vocabLocalSize_);
        Select(repeatTimesTmpL, repeatTimesUint8L, logitsInTmpL2, 1.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, vocabLocalSize_);
        Mul(logitsInTmpL, logitsInTmpL, repeatTimesTmpL, vocabLocalSize_);
        Mins(logitsInL, logitsInL, 0.0f, vocabLocalSize_);
        Duplicate(logitsInTmpL2, alpha, vocabLocalSize_);
        Select(repeatTimesTmpL, repeatTimesUint8L, logitsInTmpL2, 1.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, vocabLocalSize_);
        Mul(logitsInL, logitsInL, repeatTimesTmpL, vocabLocalSize_);
        Add(logitsInL, logitsInL, logitsInTmpL, vocabLocalSize_);
        Sub(logitsOutL, logitsInL, repeatTimesFp32L, vocabLocalSize_);

        logitsOutQueue_.EnQue(logitsOutL);
        logitsInQueue_.FreeTensor(logitsInL);
        repeatTimesInQueue_.FreeTensor(repeatTimesL);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> logitsOutL = logitsOutQueue_.DeQue<float>();
        DataCopy(logitsOutG_, logitsOutL, vocabLocalSize_);
        logitsOutQueue_.FreeTensor(logitsOutL);
    }

private:
    __gm__ int64_t *preIdsGm_;
    __gm__ float *logitsInGm_;
    __gm__ int32_t *repeatTimesGm_;
    __gm__ float *penaltyScoresGm_;
    __gm__ float *frequencyScoresGm_;
    __gm__ float *presenceScoresGm_;
    __gm__ int64_t *curLenGm_;
    __gm__ int64_t *minLenGm_;
    __gm__ int64_t *stopSeqsGm_;
    __gm__ int32_t *stopSeqsLenGm_;
    __gm__ int64_t *eosTokenIdGm_;
    __gm__ float *logitsOutGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> logitsInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> repeatTimesInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> logitsOutQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> repeatTimesOutQueue_;
    TBuf<QuePosition::VECCALC> repeatTimesFp32Buff_;
    TBuf<QuePosition::VECCALC> repeatTimesUint8Buff_;
    TBuf<QuePosition::VECCALC> repeatTimesTmpBuff_;
    TBuf<QuePosition::VECCALC> logitsTmpBuff_;
    TBuf<QuePosition::VECCALC> logitsTmpBuff2_;

    GlobalTensor<float> logitsInG_;
    GlobalTensor<float> logitsOutG_;
    GlobalTensor<int32_t> repeateTimesG_;

    int32_t vocabSize_;
    int32_t vocabLocalSize_;
    int32_t seqLen_;
    int32_t stopSeqsNum;
    int32_t stopSeqsMaxLen;
    int32_t eosLen;
    int32_t bs_;
    int32_t bsEachCore_;
};

extern "C" __global__ __aicore__ void token_penalty_multi_scores_with_stop_seqs(
    GM_ADDR preIds, GM_ADDR logits, GM_ADDR repeatTimes, GM_ADDR penaltyScores, GM_ADDR frequencyScores,
    GM_ADDR presenceScores, GM_ADDR curLen, GM_ADDR minLen, GM_ADDR stopSeqs, GM_ADDR stopSeqsLen, GM_ADDR eosTokenIds,
    GM_ADDR logitsOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TokenPenaltyMultiScoresWithStopSeqs op(
        tilingData.vs, tilingData.vsBlock, tilingData.seqLen, tilingData.stop_seqs_num, tilingData.stop_seqs_max_len, tilingData.eos_len, tilingData.bs, tilingData.bsBlock);
    op.Init(preIds, logits, repeatTimes, penaltyScores, frequencyScores, presenceScores, curLen, minLen, stopSeqs, stopSeqsLen, eosTokenIds, logitsOut);
    op.Process();
}
