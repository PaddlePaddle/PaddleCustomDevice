#include "kernel_operator.h"
using namespace AscendC;

class StepPaddle {
public:
    __aicore__ inline StepPaddle(int32_t bsz, int32_t block_size, int32_t block_num_per_seq, int32_t max_decoder_block_num, int32_t length, int32_t pre_id_length, int64_t first_token_id)
    {
        this->bsz = bsz;
        this->block_size = block_size;
        this->block_num_per_seq = block_num_per_seq;
        this->max_decoder_block_num = max_decoder_block_num;
        this->length = length;
        this->pre_id_length = pre_id_length;
        this->first_token_id = first_token_id;
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Init(__gm__ uint8_t* stop_flags, __gm__ uint8_t* seq_lens_this_time, __gm__ uint8_t* ori_seq_lens_encoder, __gm__ uint8_t* seq_lens_encoder,
                                        __gm__ uint8_t* seq_lens_decoder, __gm__ uint8_t* block_tables, __gm__ uint8_t* encoder_block_lens, __gm__ uint8_t* is_block_step,
                                        __gm__ uint8_t* step_block_list, __gm__ uint8_t* step_len, __gm__ uint8_t* recover_block_list, __gm__ uint8_t* recover_len,
                                        __gm__ uint8_t* need_block_list, __gm__ uint8_t* need_block_len, __gm__ uint8_t* used_list_len, __gm__ uint8_t* free_list,
                                        __gm__ uint8_t* free_list_len, __gm__ uint8_t* input_ids, __gm__ uint8_t* pre_ids, __gm__ uint8_t* step_idx, __gm__ uint8_t* next_tokens)
    {
        this->stop_flags = (__gm__ bool *)stop_flags;
        this->seq_lens_this_time = (__gm__ int *)seq_lens_this_time;
        this->ori_seq_lens_encoder = (__gm__ int *)ori_seq_lens_encoder;
        this->seq_lens_encoder = (__gm__ int *)seq_lens_encoder;
        this->seq_lens_decoder = (__gm__ int *)seq_lens_decoder;
        this->block_tables = (__gm__ int *)block_tables;
        this->encoder_block_lens = (__gm__ int *)encoder_block_lens;
        this->is_block_step = (__gm__ bool *)is_block_step;
        this->step_block_list = (__gm__ int *)step_block_list;
        this->step_len = (__gm__ int *)step_len;
        this->recover_block_list = (__gm__ int *)recover_block_list;
        this->recover_len = (__gm__ int *)recover_len;
        this->need_block_list = (__gm__ int *)need_block_list;
        this->need_block_len = (__gm__ int *)need_block_len;
        this->used_list_len = (__gm__ int *)used_list_len;
        this->free_list = (__gm__ int *)free_list;
        this->free_list_len = (__gm__ int *)free_list_len;
        this->input_ids = (__gm__ int64_t *)input_ids;
        this->pre_ids = (__gm__ int64_t *)pre_ids;
        this->step_idx = (__gm__ int64_t *)step_idx;
        this->next_tokens = (__gm__ int64_t *)next_tokens;
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < bsz; i++) {
            __gm__ int *block_table_now = block_tables + i * block_num_per_seq;
            if (stop_flags[i] && !is_block_step[i]) {
                const int encoder_block_len = encoder_block_lens[i];
                const int decoder_used_len = used_list_len[i];
                if (decoder_used_len > 0) {
                    pipe_barrier(PIPE_ALL);
                    const int ori_free_list_len = free_list_len[0];
                    free_list_len[0] += decoder_used_len;
                    pipe_barrier(PIPE_ALL);
                    for (int32_t j = 0; j < decoder_used_len; j++) {
                        free_list[ori_free_list_len + j] = block_table_now[encoder_block_len + j];
                        block_table_now[encoder_block_len + j] = -1;
                    }
                    encoder_block_lens[i] = 0;
                    used_list_len[i] = 0;
                    pipe_barrier(PIPE_ALL);
                }
            } else if (seq_lens_decoder[i] != 0 && block_table_now[(seq_lens_decoder[i] + 1) / block_size] == -1) {
                const int ori_need_block_len = need_block_len[0];
                need_block_len[0] = need_block_len[0] + 1;
                need_block_list[ori_need_block_len] = i;
            }
            pipe_barrier(PIPE_ALL);
        }

        pipe_barrier(PIPE_ALL);

        while (need_block_len[0] > free_list_len[0]) {
            int max_idx = 0;
            int max_used_block_num = 0;
            for (int i = 0; i < bsz; i++) {
                const int cur_block_num = is_block_step[i] ? 0 : used_list_len[i];
                if (cur_block_num > max_used_block_num) {
                    pipe_barrier(PIPE_ALL);
                    max_idx = i;
                    pipe_barrier(PIPE_ALL);
                    max_used_block_num = cur_block_num;
                    pipe_barrier(PIPE_ALL);
                }
            }

            const int encoder_block_len = encoder_block_lens[max_idx];
            __gm__ int *block_table_now = block_tables + max_idx * block_num_per_seq;
            for (int i = 0; i < max_used_block_num; i++) {
                free_list[free_list_len[0] + i] = block_table_now[encoder_block_len + i];
                block_table_now[encoder_block_len + i] = -1;
                pipe_barrier(PIPE_ALL);
            }

            step_block_list[step_len[0]] = max_idx;
            step_len[0] += 1;
            free_list_len[0] += max_used_block_num;
            stop_flags[max_idx] = true;
            is_block_step[max_idx] = true;
            seq_lens_this_time[max_idx] = 0;
            seq_lens_decoder[max_idx] = 0;
            seq_lens_encoder[max_idx] = 0;
            pipe_barrier(PIPE_ALL);
        }

        for (int32_t i = 0; i < need_block_len[0]; i++) {
            const int need_block_id = need_block_list[i];
            if (need_block_list[i] != -1) {
                if (!stop_flags[need_block_id]) {
                    used_list_len[need_block_id] += 1;
                    pipe_barrier(PIPE_ALL);
                    const int ori_free_list_len = free_list_len[0];
                    free_list_len[0] -= 1;
                    pipe_barrier(PIPE_ALL);
                    __gm__ int *block_table_now = block_tables + need_block_id * block_num_per_seq;
                    block_table_now[(seq_lens_decoder[need_block_id] + 1) / block_size] = free_list[ori_free_list_len - 1];
                }
                need_block_list[i] = -1;
                pipe_barrier(PIPE_ALL);
            }
        }

        int ori_free_list_len = free_list_len[0];
        int ori_step_len = step_len[0];
        int ori_step_block_id = step_block_list[ori_step_len - 1];
        int tmp_used_len = used_list_len[ori_step_block_id];
        int used_len = tmp_used_len < max_decoder_block_num ? tmp_used_len + 1 : tmp_used_len;
        pipe_barrier(PIPE_ALL);
        while (ori_step_len > 0 && ori_free_list_len >= used_len) {
            recover_block_list[recover_len[0]] = ori_step_block_id;
            is_block_step[ori_step_block_id] = false;
            used_list_len[ori_step_block_id] = used_len;
            ori_free_list_len -= used_len;
            step_block_list[ori_step_len - 1] = -1;
            step_len[0] -= 1;
            recover_len[0] += 1;
            ori_step_len = step_len[0];
            if (ori_step_len > 0) {
                ori_step_block_id = step_block_list[ori_step_len - 1];
                tmp_used_len = used_list_len[ori_step_block_id];
                used_len = tmp_used_len < max_decoder_block_num ? tmp_used_len + 1 : tmp_used_len;
            }
            pipe_barrier(PIPE_ALL);
        }

        need_block_len[0] = 0;

        pipe_barrier(PIPE_ALL);

        if (recover_len[0] > 0) {
            int ori_free_list_len;
            for (int32_t i = 0; i < recover_len[0]; i++) {
                const int recover_id = recover_block_list[i];
                const int ori_seq_len_encoder = ori_seq_lens_encoder[recover_id];
                const int step_idx_now = step_idx[recover_id];
                const int seq_len = ori_seq_len_encoder + step_idx_now;
                const int encoder_block_len = encoder_block_lens[recover_id];
                const int decoder_used_len = used_list_len[recover_id];
                pipe_barrier(PIPE_ALL);
                __gm__ int *block_table_now = block_tables + recover_id * block_num_per_seq;
                __gm__ int64_t *input_ids_now = input_ids + recover_id * length;
                __gm__ int64_t *pre_ids_now = pre_ids + recover_id * pre_id_length;
                pipe_barrier(PIPE_ALL);
                seq_lens_this_time[recover_id] = seq_len;
                seq_lens_encoder[recover_id] = seq_len;
                stop_flags[recover_id] = false;
                input_ids_now[ori_seq_len_encoder + step_idx_now - 1] = next_tokens[recover_id];
                input_ids_now[0] = first_token_id;
                pipe_barrier(PIPE_ALL);
                const int ori_free_list_len_0 = free_list_len[0];
                free_list_len[0] -= decoder_used_len;
                pipe_barrier(PIPE_ALL);
                ori_free_list_len = ori_free_list_len_0;
                pipe_barrier(PIPE_ALL);

                for (int32_t j = 0; j < decoder_used_len; j++) {
                    block_table_now[encoder_block_len + j] = free_list[ori_free_list_len - decoder_used_len + j];
                }

                pipe_barrier(PIPE_ALL);

                for (int32_t j = 0; j < step_idx_now - 1; j++) {
                    input_ids_now[ori_seq_len_encoder + j] = pre_ids_now[j + 1];
                }

                pipe_barrier(PIPE_ALL);
            }
            

            recover_len[0] = 0;
        }
    }

private:
    int32_t bsz = 0;
    int32_t block_size = 0;
    int32_t block_num_per_seq = 0;
    int32_t max_decoder_block_num = 0;
    int32_t length = 0;
    int32_t pre_id_length = 0;
    int64_t first_token_id = 0;

    __gm__ bool *stop_flags;
    __gm__ int *seq_lens_this_time;
    __gm__ int *ori_seq_lens_encoder;
    __gm__ int *seq_lens_encoder;
    __gm__ int *seq_lens_decoder;
    __gm__ int *block_tables;
    __gm__ int *encoder_block_lens;
    __gm__ bool *is_block_step;
    __gm__ int *step_block_list;
    __gm__ int *step_len;
    __gm__ int *recover_block_list;
    __gm__ int *recover_len;
    __gm__ int *need_block_list;
    __gm__ int *need_block_len;
    __gm__ int *used_list_len;
    __gm__ int *free_list;
    __gm__ int *free_list_len;
    __gm__ int64_t *input_ids;
    __gm__ int64_t *pre_ids;
    __gm__ int64_t *step_idx;
    __gm__ int64_t *next_tokens;
    
};

extern "C" __global__ __aicore__ void step_paddle(GM_ADDR stop_flags, GM_ADDR seq_lens_this_time, GM_ADDR ori_seq_lens_encoder, GM_ADDR seq_lens_encoder,
                                           GM_ADDR seq_lens_decoder, GM_ADDR block_tables, GM_ADDR encoder_block_lens, GM_ADDR is_block_step,
                                           GM_ADDR step_block_list, GM_ADDR step_len, GM_ADDR recover_block_list, GM_ADDR recover_len,
                                           GM_ADDR need_block_list, GM_ADDR need_block_len, GM_ADDR used_list_len, GM_ADDR free_list,
                                           GM_ADDR free_list_len, GM_ADDR input_ids, GM_ADDR pre_ids, GM_ADDR step_idx, GM_ADDR next_tokens,
                                           GM_ADDR stop_flags_out, GM_ADDR seq_lens_this_time_out, GM_ADDR seq_lens_encoder_out, GM_ADDR seq_lens_decoder_out, GM_ADDR block_tables_out,
                                           GM_ADDR encoder_block_lens_out, GM_ADDR is_block_step_out, GM_ADDR step_block_list_out, GM_ADDR step_lens_out, GM_ADDR recover_block_list_out,
                                           GM_ADDR recover_len_out, GM_ADDR need_block_list_out, GM_ADDR need_block_len_out, GM_ADDR used_list_len_out, GM_ADDR free_list_out,
                                           GM_ADDR free_list_len_out, GM_ADDR input_ids_out,
                                           GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    StepPaddle op(tilingData.bsz, tilingData.block_size, tilingData.block_num_per_seq, tilingData.max_decoder_block_num, tilingData.length, tilingData.pre_id_length, tilingData.first_token_id);
    op.Init(stop_flags, seq_lens_this_time, ori_seq_lens_encoder, seq_lens_encoder, seq_lens_decoder, block_tables, encoder_block_lens, is_block_step, step_block_list, step_len, recover_block_list, recover_len,
                                        need_block_list, need_block_len, used_list_len, free_list, free_list_len, input_ids, pre_ids, step_idx, next_tokens);
    op.Process();
}
