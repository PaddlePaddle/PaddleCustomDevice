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
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void IndexSelectKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& index,
                       int dim,
                       phi::DenseTensor* output) {
  VLOG(4) << "Call SDAA IndexSelectKernel";

  if (dim < 0) {
    dim += x.dims().size();
  }
  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT64 || index_type == phi::DataType::INT32;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  dev_ctx.template Alloc<T>(output);

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> index_dims = phi::vectorize<int>(index.dims());
  std::vector<int> output_dims = phi::vectorize<int>(output->dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t index_Desc = sdaa_ops::GetTecodnnTensorDesc(
      index_dims, index.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t output_Desc = sdaa_ops::GetTecodnnTensorDesc(
      output_dims, output->dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnIndexSelect(tecodnnHandle,
                                   dim,
                                   x_Desc,
                                   x.data(),
                                   index_Desc,
                                   index.data(),
                                   output_Desc,
                                   output->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(index_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(output_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_select,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::IndexSelectKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
