#include "kernel_operator.h"
using namespace AscendC;

class UpdateInputs {
public:
    __aicore__ inline UpdateInputs(int32_t bs, int32_t max_bsz, int32_t length)
    {
        set_atomic_none();
        set_mask_norm();
        this->lengthNum = length;
        this->bsz = bs;
        this->max_bsz = max_bsz;
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Init( __gm__ uint8_t *stop_flags, __gm__ uint8_t *not_need_stop, __gm__ uint8_t *seq_lens_this_time, 
                                 __gm__ uint8_t *seq_lens_encoder, __gm__ uint8_t *seq_lens_decoder, __gm__ uint8_t *input_ids, 
                                 __gm__ uint8_t *stop_nums, __gm__ uint8_t *is_block_step, __gm__ uint8_t *next_tokens)
    {
        this->stop_flags = (__gm__ bool *)stop_flags;
        this->not_need_stop = (__gm__ bool *)not_need_stop;
        this->seq_lens_this_time = (__gm__ int *)seq_lens_this_time;
        this->seq_lens_encoder = (__gm__ int *)seq_lens_encoder;
        this->seq_lens_decoder = (__gm__ int *)seq_lens_decoder;
        this->input_ids = (__gm__ int64_t *)input_ids;
        this->stop_nums = (__gm__ int64_t *)stop_nums;
        this->is_block_step = (__gm__ bool *)is_block_step;
        this->next_tokens = (__gm__ int64_t *)next_tokens;
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Process()
    {
        int64_t stop_sum = 0;
        __gm__ int64_t *input_ids_now;
        for (int32_t i = 0; i < max_bsz; i++) {
            if (i < bsz) {
                if (!is_block_step[i]) {
                    stop_sum += stop_flags[i] ? 1 : 0;
                }
            } else {
                stop_sum += 1;
            }
            pipe_barrier(PIPE_ALL);
            if (i < bsz) {
                seq_lens_decoder[i] = stop_flags[i] ? 0 : (seq_lens_decoder[i] == 0 ? \
                                    seq_lens_encoder[i] : seq_lens_decoder[i] + 1);
                seq_lens_this_time[i] = stop_flags[i] ? 0 : 1;
                seq_lens_encoder[i] = 0;
                input_ids_now = input_ids + i * lengthNum;
                *input_ids_now = next_tokens[i];
            }
            pipe_barrier(PIPE_ALL);
        }
        not_need_stop[0] = stop_sum < stop_nums[0];
        pipe_barrier(PIPE_ALL);
    }

private:
    __gm__ bool *stop_flags;
    __gm__ bool *not_need_stop;
    __gm__ int *seq_lens_this_time;
    __gm__ int *seq_lens_encoder;
    __gm__ int *seq_lens_decoder;
    __gm__ int64_t *input_ids;
    __gm__ int64_t *stop_nums;
    __gm__ bool *is_block_step;
    __gm__ int64_t *next_tokens;

    int32_t bsz;
    int32_t max_bsz;
    int32_t lengthNum;
};

extern "C" __global__ __aicore__ void update_inputs(GM_ADDR stop_flags, GM_ADDR not_need_stop, GM_ADDR seq_lens_this_time, 
                                                    GM_ADDR seq_lens_encoder, GM_ADDR seq_lens_decoder, GM_ADDR input_ids, 
                                                    GM_ADDR stop_nums, GM_ADDR next_tokens, GM_ADDR is_block_step,
                                                    GM_ADDR not_need_stop_out, GM_ADDR seq_lens_this_time_out, 
                                                    GM_ADDR seq_lens_encoder_out, GM_ADDR seq_lens_decoder_out, GM_ADDR input_ids_out,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    UpdateInputs op(tilingData.bs, tilingData.max_bs, tilingData.length);
    op.Init(stop_flags, not_need_stop, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, input_ids, stop_nums, is_block_step, next_tokens);
    op.Process();
}