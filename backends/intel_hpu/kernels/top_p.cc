// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <omp.h>

#include <thread>

#include "kernels/funcs.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

const int EXP_RANGE = 256;
const int SPLIT_SHIFT = 12;
const int MANTISSA_BITS = 23;

union FIConverter {
  float f;
  unsigned int i;
};

inline int get_exp(float f_num) {
  unsigned int mask = 0xFF;
  FIConverter converter;
  converter.f = f_num;
  unsigned int exp = (converter.i >> MANTISSA_BITS) & mask;

  return exp;
}

inline int split_probs(const float* f_in,
                       float* f_out,
                       int* idx_out,
                       const int length) {
  int threshold = 1 << SPLIT_SHIFT;
  int head1 = 0;
  int head2 = 0;

  int probs_dist[EXP_RANGE];
  std::memset(probs_dist, 0, sizeof(probs_dist));

  if (f_in == nullptr || f_out == nullptr || idx_out == nullptr) {
    return length;
  }

  for (int i = 0; i < length; i++) {
    int exp = get_exp(*(f_in + i));
    probs_dist[exp] += 1;
  }

  FIConverter split_f;
  int accum = 0;
  split_f.i = 0;
  for (int i = 255; i >= 0; i--) {
    accum += probs_dist[i];

    if (accum >= threshold) {
      split_f.i = ((unsigned int)i) << MANTISSA_BITS;
      break;
    }
  }

  accum = std::min(accum, length);
  head2 = accum;
  if (accum == length) {
    split_f.f = -1e30;
  }

  for (int i = 0; i < length; i++) {
    float f = *(f_in + i);
    if (f < split_f.f) {
      *(f_out + head2) = f;
      *(idx_out + head2) = i;
      head2++;
    } else {
      *(f_out + head1) = f;
      *(idx_out + head1) = i;
      head1++;
    }
  }

  return accum;
}

void handle_one_row(const float* probs,
                    const float* top_p,
                    const int32_t id,
                    const int64_t length,
                    int seed,
                    float* top_probs,
                    int64_t* top_ids) {
  std::vector<float> probs_candidates(length);
  std::vector<int> token_indices(length);
  std::mt19937 generator(seed);

  probs += length * id;
  int length2 =
      split_probs(probs, probs_candidates.data(), token_indices.data(), length);

  std::vector<int> indices(length2);
  for (int i = 0; i < length2; ++i) {
    indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(), [&](int i, int j) {
    return probs_candidates[i] > probs_candidates[j];
  });

  std::vector<float> probs_sorted(length2);
  std::vector<float> token_indices_sorted(length2);
  for (int i = 0; i < length2; i++) {
    probs_sorted[i] = probs_candidates[indices[i]];
    token_indices_sorted[i] = token_indices[indices[i]];
  }

  float prob_sum = 0.0;
  int prob_end = 0;
  float top_p_value = *(top_p + id);
  for (int i = 0; i < length2; i++) {
    if (prob_sum > top_p_value) {
      prob_end = i;
      break;
    }
    prob_sum += probs_sorted[i];
  }
  if (prob_end == 0) {
    prob_end = 1;
  }

  std::discrete_distribution<int> distribution(probs_sorted.begin(),
                                               probs_sorted.begin() + prob_end);

  auto predict = distribution(generator);
  top_ids[id] = token_indices_sorted[predict];
  top_probs[id] = probs_sorted[predict];
}

template <typename T>
void top_p_sampling(const T* probs,
                    const T* top_p,
                    const int64_t bs,
                    const int64_t length,
                    int seed,
                    T* top_probs,
                    int64_t* top_ids) {
  std::vector<std::thread> threads;
  threads.reserve(bs);

  for (int bi = 0; bi < bs; bi++) {
    threads.emplace_back(
        handle_one_row, probs, top_p, bi, length, seed, top_probs, top_ids);
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

}  // namespace custom_kernel

namespace custom_kernel {
template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& ps,
                        const paddle::optional<phi::DenseTensor>& threshold,
                        const paddle::optional<phi::DenseTensor>& topp_seed,
                        int seed,
                        int k,
                        const std::string& mode,
                        phi::DenseTensor* out,
                        phi::DenseTensor* ids,
                        phi::DenseTensor* topk_scores,
                        phi::DenseTensor* topk_ids) {
  auto x_dims = phi::vectorize<int64_t>(x.dims());
  auto meta = x.meta();
  int bs = x_dims[0];
  int length = x_dims[1];

  auto out_data = dev_ctx.template Alloc<T>(out);
  auto ids_data = dev_ctx.template Alloc<int64_t>(ids);

  top_p_sampling(const_cast<T*>(x.data<T>()),
                 (const_cast<T*>(ps.data<T>())),
                 bs,
                 length,
                 seed,
                 out_data,
                 ids_data);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    top_p_sampling, CPU, ALL_LAYOUT, custom_kernel::TopPSamplingKernel, float) {
}
