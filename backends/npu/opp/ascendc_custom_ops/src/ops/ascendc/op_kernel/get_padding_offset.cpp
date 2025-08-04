#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t ELE_PER_BLK = 8;
constexpr int32_t MAX_BATCH_NUM = 256;

namespace {

class GetPaddingOffset {
public:
    __aicore__ inline GetPaddingOffset() {}
    __aicore__ inline void Init(GM_ADDR input_ids,
                                GM_ADDR cum_offsets_now, GM_ADDR token_num,
                                GM_ADDR seq_len, GM_ADDR x_remove_padding,
                                GM_ADDR cum_offsets_out, GM_ADDR padding_offset,
                                GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k,
                                uint32_t padLength_, uint32_t batch)
    {
        this->padLength_ = padLength_;
        this->batch_ = batch;
        inputIdsGm.SetGlobalBuffer((__gm__ int64_t *)input_ids, padLength_ * batch_);
        cumOffsetsNowGm.SetGlobalBuffer((__gm__ int32_t *)cum_offsets_now, batch_);
        tokenNumGm.SetGlobalBuffer((__gm__ int64_t *)token_num, 1);
        seqLenGm.SetGlobalBuffer((__gm__ int32_t *)seq_len, batch_);
        xRemovePaddingGm.SetGlobalBuffer((__gm__ int64_t *)x_remove_padding, padLength_ * batch_);
        cumOffsetOutGm.SetGlobalBuffer((__gm__ int32_t *)cum_offsets_out, batch_);
        paddingOffsetGm.SetGlobalBuffer((__gm__ int32_t *)padding_offset, padLength_ * batch_);
        cuSeqlensQGm = (__gm__ int32_t *)cu_seqlens_q;
        cuSeqlensKGm = (__gm__ int32_t *)cu_seqlens_k;
        padLengthAlign_ = ((padLength_ + ELE_PER_BLK - 1) / ELE_PER_BLK) * ELE_PER_BLK;
        batchAlign_ = ((batch_ + ELE_PER_BLK - 1) / ELE_PER_BLK) * ELE_PER_BLK;
        pipe_.InitBuffer(inputIdsQueue_, BUFFER_NUM, padLengthAlign_ * sizeof(int64_t));
        pipe_.InitBuffer(cumOffsetsQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(seqLenQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(xRemovePaddingQueue_, BUFFER_NUM, padLengthAlign_ * sizeof(int64_t));
        pipe_.InitBuffer(cumOffsetOutQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(broadCastBuf_, padLengthAlign_ *  sizeof(int32_t));
        pipe_.InitBuffer(cumOffsetsBuf_, MAX_BATCH_NUM *  sizeof(int32_t));
        pipe_.InitBuffer(seqLenBuf_, MAX_BATCH_NUM *  sizeof(int32_t));

    }

    __aicore__ inline void Process()
    {
        pipe_barrier(PIPE_MTE2);
        pipe_barrier(PIPE_ALL);
        for (int32_t i = 0; i < batch_; i++) {
            CopyIn(i);
            CopyOnce();
            pipe_barrier(PIPE_MTE2);
            ComputeOnce(i);
            ComputeRemovePadding();
            pipe_barrier(PIPE_V);
            CopyOut(i);
            CopyOutOnce(i);
        }

        for (int32_t i = 0; i < batch_; i++) {
            int32_t cum_seq_len = (i+1) * padLength_ - *(cumOffsetOutGm.GetPhyAddr()+ i);
            *(cuSeqlensQGm + i + 1) = cum_seq_len;
            *(cuSeqlensKGm + i + 1) = cum_seq_len;
        }

    }

private:
    __aicore__ inline void CopyOnce()
    {
        LocalTensor<int32_t> seqLenLocal = seqLenQueue_.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> seqLenBufBuffer = seqLenBuf_.Get<int32_t>();
        DataCopy(seqLenLocal, seqLenGm, MAX_BATCH_NUM);
        DataCopy(cumOffsetsBuffer, cumOffsetsNowGm, MAX_BATCH_NUM);
        DataCopy(seqLenBufBuffer, seqLenGm, MAX_BATCH_NUM);
        seqLenQueue_.EnQue(seqLenLocal);
    }

    __aicore__ inline void ComputeOnce(uint32_t progress)
    {
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        LocalTensor<int32_t> seqLenLocal = seqLenQueue_.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> broadCast = broadCastBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> cumOffsetOutLocal = cumOffsetOutQueue_.AllocTensor<int32_t>();
        seqLenZero_ = seqLenLocal.GetValue(0);
        if (progress == 0) {
            Duplicate(broadCast, (int32_t)0, padLengthAlign_);
            pipe_barrier(PIPE_ALL);
        } else {
            Duplicate(broadCast, (int32_t)cumOffsetsBuffer.GetValue(progress - 1), padLengthAlign_);
            pipe_barrier(PIPE_V);
        }
        pipe_barrier(PIPE_ALL);
        cumOffsetOutLocal.SetValue(0, (int32_t)0);
        for (uint32_t x = 1; x < batch_; x++) {
                cumOffsetOutLocal.SetValue(x, cumOffsetsBuffer.GetValue(x - 1));
        }
        cumOffsetOutQueue_.EnQue<int32_t>(cumOffsetOutLocal);
        seqLenQueue_.FreeTensor(seqLenLocal);
    }

    __aicore__ inline void CopyOutOnce(uint32_t progress)
    {
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        LocalTensor<int32_t> cumOffsetOutLocal = cumOffsetOutQueue_.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> broadCast = broadCastBuf_.Get<int32_t>();
        if (progress == 0) {
            DataCopy(paddingOffsetGm, broadCast, padLengthAlign_);
        } else {
            DataCopy(paddingOffsetGm[seqLenZero_ + (progress - 1) * padLength_ -
                     cumOffsetsBuffer.GetValue(progress - 1) +
                     cumOffsetsBuffer.GetValue(0)], broadCast, padLengthAlign_);
        }
        DataCopy(cumOffsetOutGm, cumOffsetOutLocal, batchAlign_);
        cumOffsetOutQueue_.FreeTensor(cumOffsetOutLocal);
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<int64_t> inputIdsLocal = inputIdsQueue_.AllocTensor<int64_t>();
        DataCopy(inputIdsLocal, inputIdsGm[progress * padLength_], padLengthAlign_);
        inputIdsQueue_.EnQue(inputIdsLocal);
    }

    __aicore__ inline void ComputeRemovePadding()
    {
        LocalTensor<int64_t> inputIdsLocal = inputIdsQueue_.DeQue<int64_t>();
        AscendC::LocalTensor<int32_t> broadCast = broadCastBuf_.Get<int32_t>();
        AscendC::LocalTensor<int64_t> xRemovePaddingLocal = xRemovePaddingQueue_.AllocTensor<int64_t>();
        DataCopy(xRemovePaddingLocal, inputIdsLocal, padLengthAlign_);
        pipe_barrier(PIPE_ALL);
        inputIdsQueue_.FreeTensor(inputIdsLocal);
        xRemovePaddingQueue_.EnQue(xRemovePaddingLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<int32_t> cumOffsetsBuffer = cumOffsetsBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> seqLenBufBuffer = seqLenBuf_.Get<int32_t>();
        LocalTensor<int64_t> xRemovePaddingLocal = xRemovePaddingQueue_.DeQue<int64_t>();
        if (progress == 0) {
            DataCopy(xRemovePaddingGm, xRemovePaddingLocal, padLengthAlign_);
        } else {
            DataCopy(xRemovePaddingGm[progress * padLength_ - cumOffsetsBuffer.GetValue(progress - 1)],
                     xRemovePaddingLocal, padLengthAlign_);
        }
        xRemovePaddingQueue_.FreeTensor(xRemovePaddingLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM>  inputIdsQueue_, cumOffsetsQueue_, tokenNumQueue_, seqLenQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xRemovePaddingQueue_, cumOffsetOutQueue_, paddingOffsetQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cumOffsetsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> seqLenBuf_;
    GlobalTensor<int32_t> cumOffsetsNowGm, seqLenGm, cumOffsetOutGm, paddingOffsetGm;
    GlobalTensor<int64_t> inputIdsGm, tokenNumGm, xRemovePaddingGm;
    uint32_t padLength_{1};
    uint32_t batch_{1};
    uint32_t padLengthAlign_{16};
    uint32_t batchAlign_{8};
    int32_t seqLenZero_{0};
    __gm__ int32_t *cuSeqlensQGm;
    __gm__ int32_t *cuSeqlensKGm;
};
}

extern "C" __global__ __aicore__ void get_padding_offset(GM_ADDR input_ids,
    GM_ADDR cum_offsets_now, GM_ADDR token_num, GM_ADDR seq_len, GM_ADDR x_remove_padding,
    GM_ADDR cum_offsets_out, GM_ADDR padding_offset, GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GetPaddingOffset op;
    op.Init(input_ids, cum_offsets_now, token_num, seq_len,
            x_remove_padding, cum_offsets_out, padding_offset, cu_seqlens_q, cu_seqlens_k,
            tilingData.padLength, tilingData.batch);
    op.Process();
}