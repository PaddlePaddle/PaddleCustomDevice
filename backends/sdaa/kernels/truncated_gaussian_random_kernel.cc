// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
#include "paddle/phi/kernels/funcs/truncated_normal.h"

namespace custom_kernel {

template <typename T, typename Context>
void doTruncatedNormal(const Context& dev_ctx,
                       const std::vector<int>& shape,
                       float mean,
                       float std,
                       int seed,
                       phi::DataType dtype,
                       phi::DenseTensor* out) {
  VLOG(4) << "Call TruncatedNormal Op";
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t Desc =
      sdaa_ops::GetTecodnnTensorDesc(shape, dtype, TensorFormat::NHWC);
  TECODNN_CHECK(tecodnnTruncatedNormal(
      tecodnnHandle, seed, mean, std, Desc, out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

template <typename T, typename Context>
void TruncatedGaussianRandomKernel(const Context& dev_ctx,
                                   const std::vector<int>& shape,
                                   float mean,
                                   float std,
                                   int seed,
                                   phi::DataType dtype,
                                   phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA TruncatedGaussianRandomKernel";
  T* data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();

  // 1.CPU implement
  phi::DenseTensor cpu_out;
  phi::DenseTensorMeta cpu_out_meta = {out->dtype(), out->dims()};
  cpu_out.set_meta(cpu_out_meta);
  T* cpu_data = dev_ctx.template HostAlloc<T>(&cpu_out);

  double a_normal_cdf = (1.0 + erff(-2.0 / sqrtf(2.0))) / 2.0;
  double b_normal_cdf = (1.0 + erff(2.0 / sqrtf(2.0))) / 2.0;

  auto gen_sdaa = dev_ctx.GetGenerator();
  std::minstd_rand rng;
  std::uniform_real_distribution<T> dist(std::numeric_limits<T>::min(), 1);
  if (seed == 0) {
    // use global Generator seed
    auto seed_offset = gen_sdaa->IncrementOffset(1);
    uint64_t seed = seed_offset.first;
    uint64_t offset = seed_offset.second;
    rng.seed(seed);
    rng.discard(size * offset);
  } else {
    // use OP seed
    rng.seed(seed);
    rng.discard(0);
  }
  for (int64_t i = 0; i < size; i++) {
    T value = dist(rng);
    auto p = a_normal_cdf + (b_normal_cdf - a_normal_cdf) * value;
    cpu_data[i] = std::sqrt(2.0) * Erfinv(2 * p - 1) * std + mean;
  }

  // 2. CPU copy to SDAA
  phi::Copy(dev_ctx, cpu_out, dev_ctx.GetPlace(), false, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(truncated_gaussian_random,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TruncatedGaussianRandomKernel,
                          float) {}
