// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>

// #include "atb_op_runner/deprecated/atb_op_runner.h"
// #include "atb_op_runner/deprecated/set_value_by_scalar.h"
#include "glog/logging.h"
#include "paddle/extension.h"
#include "runtime/runtime.h"
// FLAGS_DEFINE_bool(npu_step_paddle_debug, true, "");

bool in_need_block_list(int seq_id,
                        int32_t* need_block_list,
                        int32_t need_block_len) {
  bool res = false;
  for (int i = 0; i < need_block_len; ++i) {
    if (seq_id == need_block_list[i]) {
      res = true;
      break;
    }
  }
  return res;
}

void free_block(const aclrtStream& stream,
                uint8_t* stop_flags_cpu,
                uint8_t* is_block_step_cpu,
                int32_t* seq_lens_this_time_cpu,
                int32_t* seq_lens_encoder_cpu,
                int32_t* seq_lens_decoder_cpu,
                int32_t* block_tables_cpu,
                int32_t* block_tables_npu,

                int32_t* encoder_block_lens,
                int32_t* used_list_len,
                int32_t* free_list,
                int32_t* free_list_len,
                int32_t* need_block_list,
                int32_t* need_block_list_len,
                int64_t* first_token_ids_cpu,
                int64_t max_batch_size,
                int64_t block_num_per_seq,
                int64_t block_size,
                bool step_max_block_flag,
                int32_t in_need_block_list_len) {
  LOG(INFO) << "<><><><><>beging free_block";
  for (int i = 0; i < max_batch_size; ++i) {
    if (i == 0) {
      step_max_block_flag = false;
      in_need_block_list_len = 0;
    }
    bool stop_flag = stop_flags_cpu[i];
    bool is_block_step = is_block_step_cpu[i];

    if (stop_flags_cpu[i] && !is_block_step_cpu[i]) {
      int encoder_block_len = encoder_block_lens[i];  // FIXME: 这里
      int decoder_used_len = used_list_len[i];
      if (decoder_used_len > 0) {
        const int ori_free_list_len = free_list_len[0];
        free_list_len[0] += decoder_used_len;
        // if (FLAGS_npu_step_paddle_debug) {
        VLOG(0) << "free block seq_id: " << i
                << ", free block num: " << decoder_used_len
                << ", encoder_block_len: " << encoder_block_len
                << ", ori_free_list_len: " << ori_free_list_len;
        // }
        for (int j = 0; j < decoder_used_len; ++j) {
          free_list[ori_free_list_len + j] =
              block_tables_cpu[i * block_num_per_seq + encoder_block_len + j];
          block_tables_cpu[i * block_num_per_seq + encoder_block_len + j] = -1;
        }
        encoder_block_lens[i] = 0;
        used_list_len[i] = 0;
        ACL_CHECK(aclrtMemcpyAsync(
            &block_tables_npu[i * block_num_per_seq + encoder_block_len],
            decoder_used_len * sizeof(int32_t),
            &block_tables_cpu[i * block_num_per_seq + encoder_block_len],
            decoder_used_len * sizeof(int32_t),
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream));
      }
    } else if (seq_lens_decoder_cpu[i] != 0 &&
               block_tables_cpu[i * block_num_per_seq +
                                (seq_lens_decoder_cpu[i] + 1) / block_size] ==
                   -1) {
      need_block_list[need_block_list_len[0]++] = i;
      // if (FLAGS_npu_step_paddle_debug) {
      VLOG(0) << "seq_id: " << i << " need block";
      // }
    }
  }
}

void dispatch_block(const aclrtStream& stream,
                    uint8_t* stop_flags_cpu,
                    uint8_t* stop_flags_npu,
                    uint8_t* is_block_step_cpu,
                    uint8_t* is_block_step_npu,
                    int32_t* seq_lens_this_time_cpu,
                    int32_t* seq_lens_this_time_npu,
                    int32_t* seq_lens_encoder_cpu,
                    int32_t* seq_lens_encoder_npu,
                    int32_t* seq_lens_decoder_cpu,
                    int32_t* seq_lens_decoder_npu,
                    int32_t* block_tables_cpu,
                    int32_t* block_tables_npu,

                    int32_t* encoder_block_lens,
                    int32_t* used_list_len,
                    int32_t* free_list,
                    int32_t* free_list_len,
                    int32_t* need_block_list,
                    int32_t* need_block_list_len,
                    int32_t* step_block_list,
                    int32_t* step_block_list_len,
                    int64_t* first_token_ids,
                    int64_t max_batch_size,
                    int64_t block_num_per_seq,
                    int64_t block_size,
                    bool step_max_block_flag,
                    int32_t in_need_block_list_len,
                    int64_t max_decoder_block_num) {
  // if (FLAGS_npu_step_paddle_debug) {
  VLOG(0) << "need_block_list_len: " << need_block_list_len[0]
          << " free_list_len: " << free_list_len[0];
  // }

  // 调度block,根据used_list_len从大到小回收block,直到满足need_block_len,
  // 已解码到最后一个block的query不发生调度
  while (need_block_list_len[0] > free_list_len[0]) {
    int seq_id = 0;
    int decoder_used_len = 0;
    for (int i = 0; i < max_batch_size; ++i) {
      int used_block_num =
          (!is_block_step_cpu[i] &&
           (step_max_block_flag || used_list_len[i] != max_decoder_block_num))
              ? used_list_len[i]
              : 0;
      if (used_block_num > decoder_used_len) {
        seq_id = i;
        decoder_used_len = used_block_num;
      }
    }
    if (decoder_used_len == 0) {
      step_max_block_flag = true;
    } else {
      int encoder_block_len = encoder_block_lens[seq_id];

      // if (FLAGS_npu_step_paddle_debug) {
      VLOG(0) << "max_id: " << seq_id << ", max_num: " << decoder_used_len
              << ", encoder_block_len: " << encoder_block_len;
      // }
      // decoder_used_len > 0
      for (int i = 0; i < decoder_used_len; ++i) {
        free_list[free_list_len[0] + i] =
            block_tables_cpu[seq_id * block_num_per_seq + encoder_block_len +
                             i];
        block_tables_cpu[seq_id * block_num_per_seq + encoder_block_len + i] =
            -1;
      }
      ACL_CHECK(aclrtMemcpyAsync(
          &block_tables_npu[seq_id * block_num_per_seq + encoder_block_len],
          decoder_used_len * sizeof(int32_t),
          &block_tables_cpu[seq_id * block_num_per_seq + encoder_block_len],
          decoder_used_len * sizeof(int32_t),
          ACL_MEMCPY_HOST_TO_DEVICE,
          stream));
      step_block_list[step_block_list_len[0]] = seq_id;
      if (in_need_block_list(seq_id,
                             need_block_list,
                             need_block_list_len[0] + in_need_block_list_len)) {
        need_block_list_len[0] -= 1;
        in_need_block_list_len += 1;
        need_block_list[seq_id] = -1;
      }
      step_block_list_len[0] += 1;
      free_list_len[0] += decoder_used_len;
      stop_flags_cpu[seq_id] = true;
      is_block_step_cpu[seq_id] = true;
      seq_lens_this_time_cpu[seq_id] = 0;
      seq_lens_decoder_cpu[seq_id] = 0;
      seq_lens_encoder_cpu[seq_id] = 0;
      ACL_CHECK(aclrtMemcpyAsync(&stop_flags_npu[seq_id],
                                 1 * sizeof(uint8_t),
                                 &stop_flags_cpu[seq_id],
                                 1 * sizeof(uint8_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream));
      ACL_CHECK(aclrtMemcpyAsync(&is_block_step_npu[seq_id],
                                 1 * sizeof(uint8_t),
                                 &is_block_step_cpu[seq_id],
                                 1 * sizeof(uint8_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream));
      ACL_CHECK(aclrtMemcpyAsync(&seq_lens_this_time_npu[seq_id],
                                 1 * sizeof(int32_t),
                                 &seq_lens_this_time_cpu[seq_id],
                                 1 * sizeof(int32_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream));
      ACL_CHECK(aclrtMemcpyAsync(&seq_lens_decoder_npu[seq_id],
                                 1 * sizeof(int32_t),
                                 &seq_lens_decoder_cpu[seq_id],
                                 1 * sizeof(int32_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream));
      ACL_CHECK(aclrtMemcpyAsync(&seq_lens_encoder_npu[seq_id],
                                 1 * sizeof(int32_t),
                                 &seq_lens_encoder_cpu[seq_id],
                                 1 * sizeof(int32_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream));
    }
  }
  // 为需要block的位置分配block，每个位置分配一个block
  for (auto i = 0; i < (need_block_list_len[0] + in_need_block_list_len); ++i) {
    if (need_block_list[i] != -1) {
      auto seq_id = need_block_list[i];
      if (!stop_flags_cpu[seq_id]) {
        used_list_len[seq_id] += 1;
        auto ori_free_list_len = free_list_len[0];
        free_list_len[0]--;
        auto block_offset = (seq_lens_decoder_cpu[seq_id] + 1) / block_size;
        block_tables_cpu[seq_id * block_num_per_seq + block_offset] =
            free_list[ori_free_list_len - 1];
        ACL_CHECK(aclrtMemcpyAsync(
            &block_tables_npu[seq_id * block_num_per_seq + block_offset],
            1 * sizeof(int32_t),
            &block_tables_cpu[seq_id * block_num_per_seq + block_offset],
            1 * sizeof(int32_t),
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream));
      }
      need_block_list[i] = -1;
    }
  }
  need_block_list_len[0] = 0;
}

void recover_block(const aclrtStream& stream,
                   uint8_t* stop_flags_cpu,
                   uint8_t* stop_flags_npu,
                   uint8_t* is_block_step_cpu,
                   uint8_t* is_block_step_npu,
                   int32_t* seq_lens_this_time_cpu,
                   int32_t* seq_lens_this_time_npu,
                   int32_t* seq_lens_encoder_cpu,
                   int32_t* seq_lens_encoder_npu,
                   int32_t* seq_lens_decoder_cpu,
                   int32_t* seq_lens_decoder_npu,
                   int32_t* block_tables_cpu,
                   int32_t* block_tables_npu,
                   int64_t* input_ids_npu,
                   int64_t* pre_ids_npu,
                   int64_t* step_idx_cpu,
                   int64_t* next_tokens_npu,
                   int64_t* first_tokens_ids_cpu,
                   int32_t* recover_block_list,
                   int32_t* recover_len,
                   int32_t* encoder_block_lens,
                   int32_t* used_list_len,
                   int32_t* free_list,
                   int32_t* free_list_len,
                   int32_t* need_block_list,
                   int32_t* need_block_list_len,
                   int32_t* step_block_list,
                   int32_t* step_block_list_len,
                   int32_t* ori_seq_lens_encoder,

                   int64_t max_batch_size,
                   int64_t max_seq_len,
                   int64_t pre_id_length,
                   int64_t block_num_per_seq,
                   int64_t block_size,
                   int64_t max_block_num) {
  // 计算可以复原的query id
  int ori_step_len = step_block_list_len[0];
  if (ori_step_len > 0) {
    int ori_free_list_len = free_list_len[0];
    int ori_step_block_id = step_block_list[ori_step_len - 1];
    int tmp_used_len = used_list_len[ori_step_block_id];
    const int max_decoder_block_num_this_seq =
        max_block_num - encoder_block_lens[ori_step_block_id];
    // 比之前调度时多分配一个block，防止马上恢复刚调度的query(比如回收的seq_id在need_block_list中）
    int used_len = tmp_used_len + 1 < max_decoder_block_num_this_seq
                       ? tmp_used_len + 1
                       : max_decoder_block_num_this_seq;
    if (ori_step_len > 0 && ori_free_list_len >= used_len) {
      // NPU 一次只复原一条数据，否则encoder warmup会失效
      // if (FLAGS_npu_step_paddle_debug) {
      VLOG(0) << "recover seq_id:" << ori_step_block_id
              << " , free_list_len: " << ori_free_list_len
              << ", used_list_len: " << used_len;
      // }
      recover_block_list[recover_len[0]] = ori_step_block_id;
      is_block_step_cpu[ori_step_block_id] = false;
      ACL_CHECK(aclrtMemcpyAsync(&is_block_step_npu[ori_step_block_id],
                                 1 * sizeof(uint8_t),
                                 &is_block_step_cpu[ori_step_block_id],
                                 1 * sizeof(uint8_t),
                                 ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream));
      used_list_len[ori_step_block_id] = used_len;
      ori_free_list_len -= used_len;
      step_block_list[ori_step_len - 1] = -1;
      step_block_list_len[0] -= 1;
      recover_len[0] += 1;
      ori_step_len = step_block_list_len[0];
      if (ori_step_len > 0) {
        ori_step_block_id = step_block_list[ori_step_len - 1];
        tmp_used_len = used_list_len[ori_step_block_id];
        used_len = tmp_used_len + 1 < max_decoder_block_num_this_seq
                       ? tmp_used_len + 1
                       : max_decoder_block_num_this_seq;
      }
    }
  }
  // if (recover_len[0] > 0 && FLAGS_npu_step_paddle_debug) {
  VLOG(0) << "recover_len: " << recover_len[0];
  // }
  for (int i = 0; i < recover_len[0]; ++i) {
    auto seq_id = recover_block_list[i];
    auto ori_seq_len_encoder = ori_seq_lens_encoder[seq_id];
    auto step_id = step_idx_cpu[seq_id];
    auto seq_len = ori_seq_len_encoder + step_id;
    auto encoder_block_len = encoder_block_lens[seq_id];
    auto decoder_used_len = used_list_len[seq_id];

    seq_lens_this_time_cpu[seq_id] = seq_len;
    seq_lens_encoder_cpu[seq_id] = seq_len;
    stop_flags_cpu[seq_id] = false;
    ACL_CHECK(aclrtMemcpyAsync(&seq_lens_this_time_npu[seq_id],
                               1 * sizeof(int32_t),
                               &seq_lens_this_time_cpu[seq_id],
                               1 * sizeof(int32_t),
                               ACL_MEMCPY_HOST_TO_DEVICE,
                               stream));
    ACL_CHECK(aclrtMemcpyAsync(&seq_lens_encoder_npu[seq_id],
                               1 * sizeof(int32_t),
                               &seq_lens_encoder_cpu[seq_id],
                               1 * sizeof(int32_t),
                               ACL_MEMCPY_HOST_TO_DEVICE,
                               stream));
    ACL_CHECK(aclrtMemcpyAsync(&stop_flags_npu[seq_id],
                               1 * sizeof(int32_t),
                               &stop_flags_cpu[seq_id],
                               1 * sizeof(int32_t),
                               ACL_MEMCPY_HOST_TO_DEVICE,
                               stream));
    auto ori_free_list_len = free_list_len[0];
    free_list_len[0] -= used_list_len[seq_id];
    // if (FLAGS_npu_step_paddle_debug) {
    VLOG(0) << "seq_id: " << seq_id
            << ", ori_seq_len_encoder: " << ori_seq_len_encoder
            << ", step_idx_now: " << step_id << ", seq_len: " << seq_len
            << ", decoder_used_len: " << decoder_used_len
            << ", ori_free_list_len_tid0: " << ori_free_list_len
            << ", free_list_len: " << free_list_len[0];
    // }
    std::memcpy(
        &block_tables_cpu[seq_id * block_num_per_seq + encoder_block_len],
        &free_list[ori_free_list_len - decoder_used_len],
        decoder_used_len * sizeof(int32_t));
    ACL_CHECK(aclrtMemcpyAsync(
        &block_tables_npu[seq_id * block_num_per_seq + encoder_block_len],
        decoder_used_len * sizeof(int32_t),
        &block_tables_cpu[seq_id * block_num_per_seq + encoder_block_len],
        decoder_used_len * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE,
        stream));
    ACL_CHECK(
        aclrtMemcpyAsync(&input_ids_npu[seq_id * max_seq_len + seq_len - 1],
                         1 * sizeof(int64_t),
                         &next_tokens_npu[seq_id],
                         1 * sizeof(int64_t),
                         ACL_MEMCPY_DEVICE_TO_DEVICE,
                         stream));
    ACL_CHECK(aclrtMemcpyAsync(&input_ids_npu[seq_id * max_seq_len],
                               1 * sizeof(int64_t),
                               &first_tokens_ids_cpu[seq_id],
                               1 * sizeof(int64_t),
                               ACL_MEMCPY_HOST_TO_DEVICE,
                               stream));
    ACL_CHECK(aclrtMemcpyAsync(
        &input_ids_npu[seq_id * max_seq_len + ori_seq_len_encoder],
        (step_id - 1) * sizeof(int64_t),
        &pre_ids_npu[seq_id * pre_id_length + 1],
        (step_id - 1) * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_DEVICE,
        stream));
  }
  recover_len[0] = 0;
}

static uint8_t* g_stop_flags_cpu = nullptr;
static uint8_t* g_is_block_step_cpu = nullptr;
static int32_t* g_seq_lens_this_time_cpu = nullptr;
static int32_t* g_seq_lens_encoder_cpu = nullptr;
static int32_t* g_seq_lens_decoder_cpu = nullptr;
static int32_t* g_block_tables_cpu = nullptr;
static int64_t* g_step_idx_cpu = nullptr;

void AtbStepPaddle(
    const paddle::Tensor& stop_flags,            // [mbs, 1]
    const paddle::Tensor& seq_lens_this_time,    // [mbs, 1]
    const paddle::Tensor& ori_seq_lens_encoder,  // cpu
    const paddle::Tensor& seq_lens_encoder,      // [mbs, 1]
    const paddle::Tensor& seq_lens_decoder,      // [mbs, 1]
    const paddle::Tensor& block_tables,          // [bsz, block_num_per_seq]
    const paddle::Tensor& encoder_block_lens,    // cpu
    const paddle::Tensor& is_block_step,         // [mbs, 1]
    const paddle::Tensor& step_block_list,       // cpu
    const paddle::Tensor& step_lens,             // cpu
    const paddle::Tensor& recover_block_list,    // cpu
    const paddle::Tensor& recover_lens,          // cpu
    const paddle::Tensor& need_block_list,       // cpu
    const paddle::Tensor& need_block_list_len,   // cpu
    const paddle::Tensor& used_list_len,         // cpu
    const paddle::Tensor& free_list,             // cpu
    const paddle::Tensor& free_list_len,         // cpu
    const paddle::Tensor& input_ids,
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& step_idx,         // [mbs, 1]
    const paddle::Tensor& next_tokens,      // [mbs, 1]
    const paddle::Tensor& first_token_ids,  // cpu, [mbs, 1]
    const int block_size,
    const int encoder_decoder_block_num) {
  auto start_point = std::chrono::high_resolution_clock::now();

  auto place = input_ids.place();
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(place));
  aclrtStream stream = reinterpret_cast<aclrtStream>(dev_ctx->stream());

  const int max_batch_size = block_tables.shape()[0];
  const int block_num_per_seq = block_tables.shape()[1];
  const int max_seq_len = input_ids.shape()[1];
  const int pre_id_length = pre_ids.shape()[1];
  const int max_decoder_block_num = pre_id_length / block_size;
  const int max_block_num = max_seq_len / block_size;

  if (!g_stop_flags_cpu) {
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&g_stop_flags_cpu),
                              max_batch_size * sizeof(uint8_t)));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&g_is_block_step_cpu),
                              max_batch_size * sizeof(uint8_t)));
    ACL_CHECK(
        aclrtMallocHost(reinterpret_cast<void**>(&g_seq_lens_this_time_cpu),
                        max_batch_size * sizeof(int32_t)));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&g_seq_lens_encoder_cpu),
                              max_batch_size * sizeof(int32_t)));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&g_seq_lens_decoder_cpu),
                              max_batch_size * sizeof(int32_t)));
    ACL_CHECK(
        aclrtMallocHost(reinterpret_cast<void**>(&g_block_tables_cpu),
                        max_batch_size * block_num_per_seq * sizeof(int32_t)));
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&g_step_idx_cpu),
                              max_batch_size * sizeof(int64_t)));
  }

  ACL_CHECK(aclrtMemcpyAsync(g_stop_flags_cpu,
                             max_batch_size * sizeof(uint8_t),
                             stop_flags.data(),
                             max_batch_size * sizeof(uint8_t),
                             ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  ACL_CHECK(aclrtMemcpyAsync(g_is_block_step_cpu,
                             max_batch_size * sizeof(uint8_t),
                             is_block_step.data(),
                             max_batch_size * sizeof(uint8_t),
                             ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  ACL_CHECK(aclrtMemcpyAsync(g_seq_lens_this_time_cpu,
                             max_batch_size * sizeof(int32_t),
                             seq_lens_this_time.data(),
                             max_batch_size * sizeof(int32_t),
                             ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  ACL_CHECK(aclrtMemcpyAsync(g_seq_lens_encoder_cpu,
                             max_batch_size * sizeof(int32_t),
                             seq_lens_encoder.data(),
                             max_batch_size * sizeof(int32_t),
                             ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  ACL_CHECK(aclrtMemcpyAsync(g_seq_lens_decoder_cpu,
                             max_batch_size * sizeof(int32_t),
                             seq_lens_decoder.data(),
                             max_batch_size * sizeof(int32_t),
                             ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  ACL_CHECK(
      aclrtMemcpyAsync(g_block_tables_cpu,
                       max_batch_size * block_num_per_seq * sizeof(int32_t),
                       block_tables.data(),
                       max_batch_size * block_num_per_seq * sizeof(int32_t),
                       ACL_MEMCPY_DEVICE_TO_HOST,
                       stream));
  dev_ctx->Wait();

  bool step_max_block_flag = false;
  int32_t in_need_block_list_len = 0;

  LOG(INFO) << "<><><><><>first_token_ids.data()" << first_token_ids.data();
  LOG(INFO) << "<><><><><>reinterpret_cast<int64_t*>(const_cast<void*>(first_"
               "token_ids.data()))<><><><><>"
            << reinterpret_cast<int64_t*>(
                   const_cast<void*>(first_token_ids.data()));

  LOG(INFO) << "<><><><><>before free block";
  free_block(
      stream,
      g_stop_flags_cpu,
      g_is_block_step_cpu,
      g_seq_lens_this_time_cpu,
      g_seq_lens_encoder_cpu,
      g_seq_lens_decoder_cpu,
      g_block_tables_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(block_tables.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(encoder_block_lens.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(used_list_len.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(free_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(free_list_len.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(need_block_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(need_block_list_len.data())),
      reinterpret_cast<int64_t*>(const_cast<void*>(first_token_ids.data())),
      max_batch_size,
      block_num_per_seq,
      block_size,
      step_max_block_flag,
      in_need_block_list_len);
  LOG(INFO) << "<><><><><>after free block";

  dispatch_block(
      stream,
      g_stop_flags_cpu,
      reinterpret_cast<uint8_t*>(const_cast<void*>(stop_flags.data())),
      g_is_block_step_cpu,
      reinterpret_cast<uint8_t*>(const_cast<void*>(is_block_step.data())),
      g_seq_lens_this_time_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(seq_lens_this_time.data())),
      g_seq_lens_encoder_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(seq_lens_encoder.data())),
      g_seq_lens_decoder_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(seq_lens_decoder.data())),
      g_block_tables_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(block_tables.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(encoder_block_lens.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(used_list_len.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(free_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(free_list_len.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(need_block_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(need_block_list_len.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(step_block_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(step_lens.data())),
      reinterpret_cast<int64_t*>(const_cast<void*>(first_token_ids.data())),
      max_batch_size,
      block_num_per_seq,
      block_size,
      step_max_block_flag,
      in_need_block_list_len,
      max_decoder_block_num);

  ACL_CHECK(aclrtMemcpyAsync(g_step_idx_cpu,
                             max_batch_size * sizeof(int64_t),
                             step_idx.data(),
                             max_batch_size * sizeof(int64_t),
                             ACL_MEMCPY_DEVICE_TO_HOST,
                             stream));
  dev_ctx->Wait();
  recover_block(
      stream,
      g_stop_flags_cpu,
      reinterpret_cast<uint8_t*>(const_cast<void*>(stop_flags.data())),
      g_is_block_step_cpu,
      reinterpret_cast<uint8_t*>(const_cast<void*>(is_block_step.data())),
      g_seq_lens_this_time_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(seq_lens_this_time.data())),
      g_seq_lens_encoder_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(seq_lens_encoder.data())),
      g_seq_lens_decoder_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(seq_lens_decoder.data())),
      g_block_tables_cpu,
      reinterpret_cast<int32_t*>(const_cast<void*>(block_tables.data())),
      reinterpret_cast<int64_t*>(const_cast<void*>(input_ids.data())),
      reinterpret_cast<int64_t*>(const_cast<void*>(pre_ids.data())),
      g_step_idx_cpu,
      reinterpret_cast<int64_t*>(const_cast<void*>(next_tokens.data())),
      reinterpret_cast<int64_t*>(const_cast<void*>(first_token_ids.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(recover_block_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(recover_lens.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(encoder_block_lens.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(used_list_len.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(free_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(free_list_len.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(need_block_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(need_block_list_len.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(step_block_list.data())),
      reinterpret_cast<int32_t*>(const_cast<void*>(step_lens.data())),
      reinterpret_cast<int32_t*>(
          const_cast<void*>(ori_seq_lens_encoder.data())),
      max_batch_size,
      max_seq_len,
      pre_id_length,
      block_num_per_seq,
      block_size,
      max_block_num);

  dev_ctx->Wait();

  auto end_point = std::chrono::high_resolution_clock::now();
}

PD_BUILD_OP(step_paddle_op)
    .Inputs({"stop_flags",
             "seq_lens_this_time",
             "ori_seq_lens_encoder",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables",
             "encoder_block_lens",
             "is_block_step",
             "step_block_list",
             "step_lens",
             "recover_block_list",
             "recover_lens",
             "need_block_list",
             "need_block_len",
             "used_list_len",
             "free_list",
             "free_list_len",
             "input_ids",
             "pre_ids",
             "step_idx",
             "next_tokens",
             "first_tokens_ids"})
    .Attrs({"block_size: int", "encoder_decoder_block_num: int"})
    .Outputs({"stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "block_tables_out",
              "encoder_block_lens_out",
              "is_block_step_out",
              "step_block_list_out",
              "step_lens_out",
              "recover_block_list_out",
              "recover_lens_out",
              "need_block_list_out",
              "need_block_len_out",
              "used_list_len_out",
              "free_list_out",
              "free_list_len_out",
              "input_ids_out",
              "first_token_ids_out"})
    .SetInplaceMap({{"stop_flags", "stop_flags_out"},                  // npu
                    {"seq_lens_this_time", "seq_lens_this_time_out"},  // npu
                    {"seq_lens_encoder", "seq_lens_encoder_out"},      // npu
                    {"seq_lens_decoder", "seq_lens_decoder_out"},      // npu
                    {"block_tables", "block_tables_out"},              // npu
                    {"encoder_block_lens", "encoder_block_lens_out"},  // cpu
                    {"is_block_step", "is_block_step_out"},            // npu
                    {"step_block_list", "step_block_list_out"},
                    {"step_lens", "step_lens_out"},
                    {"recover_block_list", "recover_block_list_out"},
                    {"recover_lens", "recover_lens_out"},
                    {"need_block_list", "need_block_list_out"},
                    {"need_block_len", "need_block_len_out"},
                    {"used_list_len", "used_list_len_out"},
                    {"free_list", "free_list_out"},
                    {"free_list_len", "free_list_len_out"},
                    {"input_ids", "input_ids_out"},
                    {"first_tokens_ids", "first_token_ids_out"}})
    .SetKernelFn(PD_KERNEL(AtbStepPaddle));
