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
#include "kernels/funcs/slice_utils.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename Context>
void doNonZeroTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* nonzeroCount,
                     phi::DenseTensor* out) {
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  bool as_tuple = false;

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc;
  if (x.dtype() == phi::DataType::BOOL) {
    x_Desc =
        sdaa_ops::GetTecodnnBoolTensorDesc(x_dims, TensorFormat::Undefined);
  } else {
    x_Desc = sdaa_ops::GetTecodnnTensorDesc(
        x_dims, x.dtype(), TensorFormat::Undefined);
  }
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnNonZero(tecodnnHandle,
                               as_tuple,
                               x_Desc,
                               x.data(),
                               out_Desc,
                               out->data(),
                               nonzeroCount->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void NonZeroKernel(const Context& dev_ctx,
                   const phi::DenseTensor& condition,
                   phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA NonZeroKernel";

  int64_t numel = condition.numel();
  int64_t rank = condition.dims().size();

  phi::DenseTensor out_temp, nonzeroCount;
  out_temp.Resize(phi::make_ddim({numel, rank}));
  dev_ctx.template Alloc<int64_t>(&out_temp);

  nonzeroCount.Resize(phi::make_ddim({1}));
  dev_ctx.template Alloc<int64_t>(&nonzeroCount);

  custom_kernel::doNonZeroTensor<Context>(
      dev_ctx, condition, &nonzeroCount, &out_temp);

  phi::DenseTensor nonzeroCountHost;
  phi::Copy(dev_ctx, nonzeroCount, phi::CPUPlace(), true, &nonzeroCountHost);
  auto nonzeroNum = *nonzeroCountHost.data<int64_t>();

  out->Resize(phi::make_ddim({nonzeroNum, rank}));
  dev_ctx.template Alloc<int64_t>(out);

  if (nonzeroNum == 0) {
    return;
  }
  auto out_tensor = custom_kernel::Slice(
      out_temp, static_cast<int64_t>(0), static_cast<int64_t>(0) + nonzeroNum);
  phi::Copy(dev_ctx, out_tensor, out->place(), false, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(nonzero,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::NonZeroKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
