#include "kernel_operator.h"
using namespace AscendC;

namespace {

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t ELE_PER_BLK = 16;
constexpr int32_t MAX_BATCH_NUM = 256;

class RebuildPadding {
public:
    __aicore__ inline RebuildPadding(int32_t bs, int32_t dim_embed, int32_t token_num,int32_t max_input_length)
    {
        this->bs_ = bs;
        this->dimEmbed_ = dim_embed;
        this->tokenNum_ = token_num;
        this->maxInputLength_ = max_input_length; 
    }

    __aicore__ inline void Init(GM_ADDR tmpOut,
                                GM_ADDR cum_offsets, GM_ADDR seq_lens_decoder,
                                GM_ADDR seq_lens_encoder, GM_ADDR out)
    {
        dimEmbedAlign_ = (dimEmbed_ + ELE_PER_BLK - 1) / ELE_PER_BLK * ELE_PER_BLK;
        tmpOutGm.SetGlobalBuffer((__gm__ half *)tmpOut, tokenNum_ * dimEmbed_);
        cumOffsetsGm.SetGlobalBuffer((__gm__ int32_t *)cum_offsets, bs_);
        seqLensDecoderGm.SetGlobalBuffer((__gm__ int32_t *)seq_lens_decoder, bs_);
        seqLensEncoderGm.SetGlobalBuffer((__gm__ int32_t *)seq_lens_encoder, bs_);
        outGm.SetGlobalBuffer((__gm__ half *)out, bs_ * dimEmbed_);
        
        pipe_.InitBuffer(tmpOutQueue_, BUFFER_NUM, dimEmbedAlign_ * sizeof(half));
        pipe_.InitBuffer(cumOffsetsQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(seqLensDecoderQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(seqLensEncoderQueue_, BUFFER_NUM, MAX_BATCH_NUM * sizeof(int32_t));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, dimEmbedAlign_ * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < bs_; i++) {
            CopyOnce();
            pipe_barrier(PIPE_ALL);
            CopyIn(i);
            pipe_barrier(PIPE_ALL);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyOnce()
    {
        LocalTensor<int32_t> cumOffsetsLocal = cumOffsetsQueue_.AllocTensor<int32_t>();
        pipe_barrier(PIPE_ALL);
        DataCopy(cumOffsetsLocal, cumOffsetsGm, MAX_BATCH_NUM);
        cumOffsetsQueue_.EnQue(cumOffsetsLocal);

        LocalTensor<int32_t> seqLensDecoderLocal = seqLensDecoderQueue_.AllocTensor<int32_t>();
        pipe_barrier(PIPE_ALL);
        DataCopy(seqLensDecoderLocal, seqLensDecoderGm, MAX_BATCH_NUM);
        seqLensDecoderQueue_.EnQue(seqLensDecoderLocal);

        LocalTensor<int32_t> seqLensEncoderLocal = seqLensEncoderQueue_.AllocTensor<int32_t>();
        pipe_barrier(PIPE_ALL);
        DataCopy(seqLensEncoderLocal, seqLensEncoderGm, MAX_BATCH_NUM);
        seqLensEncoderQueue_.EnQue(seqLensEncoderLocal);
    }
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<half> tmpOutLocal = tmpOutQueue_.AllocTensor<half>();
        LocalTensor<int32_t> cumOffsetsLocal = cumOffsetsQueue_.DeQue<int32_t>();
        LocalTensor<int32_t> seqLensDecoderLocal = seqLensDecoderQueue_.DeQue<int32_t>();
        LocalTensor<int32_t> seqLensEncoderLocal = seqLensEncoderQueue_.DeQue<int32_t>();
        int32_t decoderVal = seqLensDecoderLocal.GetValue(progress);
        int32_t encoderVal = seqLensEncoderLocal.GetValue(progress);
        int32_t cumOffset = cumOffsetsLocal.GetValue(progress);
        pipe_barrier(PIPE_ALL);
        
        if (decoderVal == 0) {
            if (encoderVal != 0) {
                tempVal_ = progress * maxInputLength_ - cumOffset + (encoderVal - 1);
                pipe_barrier(PIPE_ALL);
                DataCopy(tmpOutLocal, tmpOutGm[tempVal_ * dimEmbed_], dimEmbedAlign_);
            }
        } else {
            tempVal_ = progress * maxInputLength_ - cumOffset;
            pipe_barrier(PIPE_ALL);
            DataCopy(tmpOutLocal, tmpOutGm[tempVal_ * dimEmbed_], dimEmbedAlign_);
        }
        tmpOutQueue_.EnQue(tmpOutLocal);
        cumOffsetsQueue_.FreeTensor(cumOffsetsLocal);
        seqLensDecoderQueue_.FreeTensor(seqLensDecoderLocal);
        seqLensEncoderQueue_.FreeTensor(seqLensEncoderLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<half> tmpOutLocal = tmpOutQueue_.DeQue<half>();
        pipe_barrier(PIPE_ALL);
        DataCopy(outGm[progress * dimEmbed_], tmpOutLocal, dimEmbedAlign_);
        tmpOutQueue_.FreeTensor(tmpOutLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> tmpOutQueue_, cumOffsetsQueue_, seqLensDecoderQueue_, seqLensEncoderQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    GlobalTensor<half> tmpOutGm, outGm;
    GlobalTensor<int32_t> cumOffsetsGm, seqLensDecoderGm, seqLensEncoderGm;
    int32_t bs_{1};
    int32_t dimEmbed_{16};
    int32_t dimEmbedAlign_{16};
    int32_t maxInputLength_{64};
    int32_t tempVal_ = 0;
    int32_t tokenNum_ = 0;
};
}

extern "C" __global__ __aicore__ void rebuild_padding(GM_ADDR tmp_out, GM_ADDR cum_offsets, GM_ADDR seq_lens_decoder,
    GM_ADDR seq_lens_encoder, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    RebuildPadding op(tilingData.bs, tilingData.dim_embed, tilingData.token_num,tilingData.max_input_length);
    op.Init(tmp_out, cum_offsets, seq_lens_decoder, seq_lens_encoder, out);
    op.Process();
}