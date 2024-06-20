// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/nv_align.h"
#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void DropoutNVAlign(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const paddle::optional<phi::DenseTensor>& seed_tensor,
                    float p,
                    int seed,
                    bool fix_seed,
                    const char* mode,
                    phi::DenseTensor* out,
                    phi::DenseTensor* mask) {
  // Align sdaa with NV device
  uint64_t seed_data;
  uint64_t increment;
  int max_threads, sm_count;
  int threads, blocks;
  size_t size = phi::product(mask->dims());

  // only float and half, so vec_size is 4
  constexpr int vec_size = 4;

  custom_kernel::GetGPUConfig(mode, &max_threads, &sm_count);
  custom_kernel::GetBlockGrid(
      size, max_threads, sm_count, vec_size, &threads, &blocks);

  size_t max_grid_size = max_threads * sm_count / threads;
  size_t grid_size = std::min(static_cast<size_t>(blocks), max_grid_size);
  auto offset = ((size - 1) / (grid_size * threads * vec_size) + 1) * vec_size;

  custom_kernel::GetSeed(
      dev_ctx, seed_tensor, seed, fix_seed, offset, &seed_data, &increment);
  VLOG(4) << "dropout: size=" << size << ", vec_size=" << vec_size
          << ", block_size=" << threads << ", grid_size=" << grid_size
          << ", seed=" << seed_data << ", offset=" << increment
          << ", increment=" << offset;
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);

  phi::DenseTensor x_temp, out_temp;
  x_temp = x;
  out_temp = *out;

  TCUS_CHECK(sdcops::pd_dropout_kernel(x_temp.data(),
                                       out_temp.data(),
                                       mask->data(),
                                       sdaa_ops::ToExtendDataType(x.dtype()),
                                       size,
                                       1 - p,
                                       seed_data,
                                       increment,
                                       grid_size * threads,
                                       custom_stream));
}

template <typename T, typename Context>
void DropoutKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const paddle::optional<phi::DenseTensor>& seed_tensor,
                   const phi::Scalar& p,
                   bool is_test,
                   const std::string& mode,
                   int seed,
                   bool fix_seed,
                   phi::DenseTensor* out,
                   phi::DenseTensor* mask) {
  VLOG(4) << "Call SDAA DropoutKernel";

  dev_ctx.template Alloc<T>(out);

  // check arguments
  const bool is_upscale = (mode == "upscale_in_train");
  PADDLE_ENFORCE_EQ(is_upscale,
                    true,
                    phi::errors::InvalidArgument(
                        "tecodnn only support mode is upscale_in_train"));

  if (is_test) {
    // dropout op for inference: out = input;
    VLOG(4) << "[Dropout] upscale_in_train test mode, copy out.";
    phi::Copy(dev_ctx, x, out->place(), false, out);
    return;
  }

  dev_ctx.template Alloc<uint8_t>(mask);
  auto* mask_data = mask->data<uint8_t>();

  float dropout_prob = p.to<float>();

  if (dropout_prob == 1.0f) {
    sdaa_ops::doFillTensor(dev_ctx, static_cast<T>(0), out->dtype(), out);
    sdaa_ops::doFillTensor(
        dev_ctx, static_cast<uint8_t>(0), mask->dtype(), mask);
    return;
  }

  if (dropout_prob == 0.f) {
    phi::Copy(dev_ctx, x, out->place(), false, out);
    sdaa_ops::doFillTensor(
        dev_ctx, static_cast<uint8_t>(1), mask->dtype(), mask);
    return;
  }

  const char* value = std::getenv(ALIGN_NV);
  if (value) {
    DropoutNVAlign<T>(dev_ctx,
                      x,
                      seed_tensor,
                      dropout_prob,
                      seed,
                      fix_seed,
                      value,
                      out,
                      mask);
    return;
  }

  int seed_number = 0;
  if (seed_tensor) {
    std::vector<int> seeds;
    TensorToVector(dev_ctx, seed_tensor.get(), dev_ctx, &seeds);
    seed_number = seeds[0];
  } else if (!fix_seed) {
    auto& engine = *dev_ctx.GetGenerator()->GetCPUEngine();
    seed_number = static_cast<int>(engine());
  } else {
    seed_number = seed;
  }

  // setup descriptors
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t xy_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(x.dims()), x.dtype(), TensorFormat::NHWC);
  tecodnnDropoutDescriptor_t dropout_Desc;
  TECODNN_CHECK(tecodnnCreateDropoutDescriptor(&dropout_Desc));

  // set states
  size_t act_statesSize = 4 * 1024 * sizeof(int);
  TECODNN_CHECK(tecodnnDropoutGetStatesSize(tecodnnHandle, &act_statesSize));
  phi::DenseTensorMeta meta = {phi::DataType::INT8,
                               {static_cast<int>(act_statesSize)}};

  phi::DenseTensor states;
  states.set_meta(meta);
  dev_ctx.template Alloc<int8_t>(&states);

  size_t reserveSpaceSize = 0;
  TECODNN_CHECK(tecodnnDropoutGetReserveSpaceSize(xy_Desc, &reserveSpaceSize));

  // execute dropout forward
  TECODNN_CHECK(tecodnnSetDropoutDescriptor(dropout_Desc,
                                            tecodnnHandle,
                                            dropout_prob,
                                            states.data(),
                                            act_statesSize,
                                            seed_number));
  TECODNN_CHECK(tecodnnDropoutForward(tecodnnHandle,
                                      dropout_Desc,
                                      xy_Desc,
                                      x.data<T>(),
                                      xy_Desc,
                                      out->data<T>(),
                                      mask->data<uint8_t>(),
                                      reserveSpaceSize));

  // destroy descriptors
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(xy_Desc));
  TECODNN_CHECK(tecodnnDestroyDropoutDescriptor(dropout_Desc));
}

template <typename T, typename Context>
void DropoutGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& mask,
                       const phi::DenseTensor& dout,
                       const phi::Scalar& p,
                       bool is_test,
                       const std::string& mode,
                       phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA DropoutGradKernel";

  PADDLE_ENFORCE_EQ(
      is_test,
      false,
      phi::errors::InvalidArgument(
          "Dropout GradOp is only callable when is_test is false"));

  // basic settings
  dev_ctx.template Alloc<T>(dx);
  int seed_number = 0;
  phi::DDim x_dims = dout.dims();
  float dropout_prob = p.to<float>();

  // setup basic descriptors
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t xy_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(x_dims), dout.dtype(), TensorFormat::NHWC);
  tecodnnDropoutDescriptor_t dropout_Desc;
  TECODNN_CHECK(tecodnnCreateDropoutDescriptor(&dropout_Desc));

  // set states
  size_t act_statesSize = 4 * 1024 * sizeof(int);
  TECODNN_CHECK(tecodnnDropoutGetStatesSize(tecodnnHandle, &act_statesSize));
  phi::DenseTensorMeta meta = {phi::DataType::INT8,
                               {static_cast<int>(act_statesSize)}};
  phi::DenseTensor states;
  states.set_meta(meta);
  dev_ctx.template Alloc<int8_t>(&states);

  size_t reserveSpaceSize = 0;
  TECODNN_CHECK(tecodnnDropoutGetReserveSpaceSize(xy_Desc, &reserveSpaceSize));
  void* mask_void = const_cast<void*>(mask.data());

  // execute dropout backward
  TECODNN_CHECK(tecodnnRestoreDropoutDescriptor(dropout_Desc,
                                                tecodnnHandle,
                                                dropout_prob,
                                                states.data(),
                                                act_statesSize,
                                                seed_number));
  TECODNN_CHECK(tecodnnDropoutBackward(tecodnnHandle,
                                       dropout_Desc,
                                       xy_Desc,
                                       dout.data<T>(),
                                       xy_Desc,
                                       dx->data<T>(),
                                       mask_void,
                                       reserveSpaceSize));

  // destroy descriptors
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(xy_Desc));
  TECODNN_CHECK(tecodnnDestroyDropoutDescriptor(dropout_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(dropout,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::DropoutKernel,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}

PD_REGISTER_PLUGIN_KERNEL(dropout_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::DropoutGradKernel,
                          phi::dtype::float16,
                          float) {}
