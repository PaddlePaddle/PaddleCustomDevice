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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void doGatherTensor(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& index,
                    const phi::Scalar& axis,
                    phi::DenseTensor* out) {
  int axis_ = axis.to<int32_t>();
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> index_dims = phi::vectorize<int>(index.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  tecodnnTensorDescriptor_t xDesc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t indexDesc = sdaa_ops::GetTecodnnTensorDesc(
      index_dims, index.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t outDesc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  TECODNN_CHECK(tecodnnGather(tecodnnHandle,
                              axis_,
                              xDesc,
                              x.data(),
                              indexDesc,
                              index.data(),
                              outDesc,
                              out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(xDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(indexDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(outDesc));
}

template <typename T, typename Context>
void GatherKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& index,
                  const phi::Scalar& axis,
                  phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA GatherKernel";

  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0) return;
  doGatherTensor<T, Context>(dev_ctx, x, index, axis, out);
}

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& index,
                      const phi::DenseTensor& dout,
                      const phi::Scalar& axis,
                      phi::DenseTensor* dx) {
  VLOG(4) << "CALL SDAA GatherGradKernel";

  dev_ctx.template Alloc<T>(dx);

  PADDLE_ENFORCE_EQ(
      axis.to<int32_t>(),
      0,
      phi::errors::Unimplemented(
          "sdaa only support axis=0 in gather_grad kernel, but got %s",
          axis.to<int32_t>()));

  const auto index_dims = index.dims();
  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of index should be 1 when it is 2D "
                          "but we get %d",
                          index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(index_dims.size() == 1 || index_dims.size() == 0,
                      true,
                      phi::errors::InvalidArgument(
                          "The index should be 0D or 1D, when it is not 2D, "
                          "but we get %d",
                          index_dims.size()));
  }

  phi::DenseTensor zeroslike_x;
  phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
  zeroslike_x.set_meta(meta);
  dev_ctx.template Alloc<T>(&zeroslike_x);

  auto value = static_cast<T>(0);

  sdaa_ops::doFillTensor<T>(dev_ctx, value, x.dtype(), &zeroslike_x);

  sdaa_ops::doScatterTensor(dev_ctx, zeroslike_x, index, dout, false, dx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GatherKernel,
                          double,
                          float,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gather_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GatherGradKernel,
                          float,
                          phi::dtype::float16) {}
