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

#include "kernels/funcs/nv_align.h"
#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {

static constexpr char kDefaultRandMode[] = "V100";

template <typename T, typename Context>
void RandintKernelNVAlign(const Context& dev_ctx,
                          int low,
                          int high,
                          const phi::IntArray& shape,
                          phi::DataType dtype,
                          phi::DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);

  bool need_trans = dtype == phi::DataType::INT64;

  // Align sdaa with NV device
  uint64_t seed_data;
  uint64_t increment;
  int max_threads, sm_count;
  int threads, blocks;
  size_t size = out->numel();

  // only float and half, so vec_size is 4
  constexpr int vec_size = 4;
  const char* mode = kDefaultRandMode;
  if (std::getenv(ALIGN_NV)) {
    mode = std::getenv(ALIGN_NV);
  }

  custom_kernel::GetGPUConfig(mode, &max_threads, &sm_count);
  custom_kernel::GetBlockGrid(
      size, max_threads, sm_count, vec_size, &threads, &blocks);

  size_t max_grid_size = max_threads * sm_count / threads;
  size_t grid_size = std::min(static_cast<size_t>(blocks), max_grid_size);
  auto offset = ((size - 1) / (grid_size * threads * vec_size) + 1) * vec_size;

  custom_kernel::GetSeed(
      dev_ctx, paddle::none, 0, false, offset, &seed_data, &increment);
  // VLOG(4) << "randint: size=" << size << ", vec_size=" << vec_size
  //         << ", block_size=" << threads << ", grid_size=" << grid_size
  //         << ", seed=" << seed_data << ", offset=" << increment
  //         << ", increment=" << offset;

  phi::DenseTensor out_int32{};
  if (need_trans) {
    auto out_meta = phi::DenseTensorMeta{phi::DataType::INT32, out->dims()};
    out_int32.set_meta(out_meta);
    dev_ctx.template Alloc<int>(&out_int32);
  }
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  TCUS_CHECK(sdcops::random_int_from_to_kernel(
      need_trans ? out_int32.data() : out->data(),
      out->numel(),
      high - low,
      low,
      seed_data,
      increment,
      custom_stream))

  if (need_trans) {
    sdaa_ops::doCastTensor(dev_ctx, out_int32, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(randint,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::RandintKernelNVAlign,
                          int,
                          int64_t) {}
