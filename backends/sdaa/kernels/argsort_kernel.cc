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

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <thread>

#include "kernels/funcs/sdaa_baseop.h"
#include "tecodnn.h"  // NOLINT
namespace custom_kernel {

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const phi::DenseTensor& in,
                   int axis,
                   bool descending,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {
  VLOG(4) << "call sdaa ArgsortKernel";
  PADDLE_ENFORCE_EQ(
      descending,
      true,
      phi::errors::InvalidArgument("tecodnn only support descending = true."
                                   "But recived: descending is %d",
                                   descending));
  int dim_size = in.dims().size();
  PADDLE_ENFORCE_EQ(
      dim_size,
      1,
      phi::errors::InvalidArgument("tecodnn only support input is 1 D."
                                   "But recived: input dims is %d",
                                   dim_size));
  const int Len = indices->numel();
  bool flag = (Len >= 4096) && (Len == (-Len & Len));
  PADDLE_ENFORCE_EQ(
      flag,
      true,
      phi::errors::InvalidArgument("Customized for shape should be the "
                                   "integral power of 2 and not less than 4096."
                                   "But recived: shape is %d",
                                   Len));
  dev_ctx.template Alloc<T>(output);
  dev_ctx.template Alloc<int64_t>(indices);
  std::vector<int> ind_dimensions(1, Len);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      ind_dimensions, in.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t index_Desc = sdaa_ops::GetTecodnnTensorDesc(
      ind_dimensions, indices->dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      ind_dimensions, output->dtype(), TensorFormat::Undefined);
  void* index = NULL;
  // Customized for the shape of ppocr-det model, index param will be ignored.
  tecodnnCustomArgsort(tecodnnHandle,
                       -1,
                       descending,
                       x_Desc,
                       in.data(),
                       index_Desc,
                       index,
                       y_Desc,
                       output->data());
  int64_t start = 0;
  int64_t end = indices->numel();
  int64_t step = 1;
  tecodnnTensorDescriptor_t indices_Desc = sdaa_ops::GetTecodnnTensorDesc(
      ind_dimensions, indices->dtype(), TensorFormat::Undefined);
  // set indices to range(0, Len), to solve core dump error for argsort_grad!
  TECODNN_CHECK(tecodnnArange(
      tecodnnHandle, &start, &end, &step, indices_Desc, indices->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(index_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(indices_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    argsort, sdaa, ALL_LAYOUT, custom_kernel::ArgsortKernel, float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
