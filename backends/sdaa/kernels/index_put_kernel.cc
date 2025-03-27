// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

namespace custom_kernel {

template <typename T, typename Context>
void doArangeTensor(const Context& dev_ctx,
                    const T& start,
                    const T& end,
                    const T& step,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<const phi::DenseTensor*>& indices,
                    const phi::DenseTensor& value,
                    bool accumulate,
                    phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA IndexPutKernel.";

  if (!out->initialized()) {
    dev_ctx.template Alloc<T>(out);
    // tecodnnIndexPut only support inplace for now
    TensorCopy(dev_ctx, x, false, out);
  }

  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      phi::errors::InvalidArgument(
          "The data type of tensor value must be same to the data type "
          "of tensor x."));

  PADDLE_ENFORCE_EQ(indices.empty(),
                    false,
                    phi::errors::InvalidArgument("Indices cannot be empty."));

  std::vector<phi::DenseTensor> tmp_args;
  std::vector<const phi::DenseTensor*> int_indices_v =
      custom_kernel::DealWithBoolIndices<T, Context>(
          dev_ctx, indices, &tmp_args);

  if (int_indices_v.empty()) {
    return;
  }

  auto bd_dim = custom_kernel::BroadCastTensorsDims(int_indices_v);

  std::vector<int64_t> res_dim_v(phi::vectorize(bd_dim));
  std::vector<const phi::DenseTensor*> res_indices_v(x.dims().size(), nullptr);
  std::vector<phi::DenseTensor> tmp_res_indices_v;
  std::vector<phi::DenseTensor> range_tensor_v;

  for (int i = static_cast<int>(int_indices_v.size()); i < x.dims().size();
       ++i) {
    phi::DenseTensor range_tensor;
    range_tensor.Resize(phi::make_ddim({x.dims()[i]}));
    dev_ctx.template Alloc<int64_t>(&range_tensor);
    custom_kernel::doArangeTensor<int64_t, Context>(
        dev_ctx, 0, x.dims()[i], 1, &range_tensor);
    range_tensor_v.emplace_back(range_tensor);
  }

  custom_kernel::DealWithIndices<T, Context>(dev_ctx,
                                             x,
                                             int_indices_v,
                                             &res_indices_v,
                                             &tmp_res_indices_v,
                                             range_tensor_v,
                                             bd_dim,
                                             &res_dim_v);
  phi::DenseTensor value_tmp;
  if (value.numel() != 1) {
    phi::DenseTensorMeta meta = {value.dtype(), phi::make_ddim(res_dim_v)};
    value_tmp.set_meta(meta);

    custom_kernel::ExpandKernel<T, Context>(
        dev_ctx, value, phi::IntArray(res_dim_v), &value_tmp);
  } else {
    value_tmp = value;
  }

  std::vector<tecodnnTensorDescriptor_t> indicesDesc;
  std::vector<const void*> tensor_list(res_indices_v.size());
  for (size_t i = 0; i < res_indices_v.size(); ++i) {
    tensor_list[i] = res_indices_v[i]->data();
    // tecodnnIndexPut only support 1-D indices when indices dtype is int64
    tecodnnTensorDescriptor_t indexDesc = sdaa_ops::GetTecodnnTensorDesc(
        std::vector<int>{static_cast<int>(res_indices_v[i]->numel())},
        res_indices_v[i]->dtype(),
        TensorFormat::Undefined);
    indicesDesc.emplace_back(indexDesc);
  }

  phi::DenseTensor index;
  int64_t index_size = res_indices_v.size() * sizeof(void*);
  index.Resize({index_size});
  dev_ctx.template Alloc<int8_t>(&index);
  AsyncMemCpyH2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 index.data(),
                 tensor_list.data(),
                 index_size);

  tecodnnHandle_t tecodnnHanndle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t valueDesc = sdaa_ops::GetTecodnnTensorDesc(
      std::vector<int>{static_cast<int>(value_tmp.numel())},
      value_tmp.dtype(),
      TensorFormat::Undefined);
  tecodnnTensorDescriptor_t inputDesc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(x.dims()), x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t outputDesc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(out->dims()), out->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnIndexPut(tecodnnHanndle,
                                res_indices_v.size(),
                                accumulate,
                                indicesDesc.data(),
                                reinterpret_cast<void**>(index.data()),
                                valueDesc,
                                value_tmp.data(),
                                inputDesc,
                                x.data(),
                                outputDesc,
                                out->data()));
  for (size_t i = 0; i < indicesDesc.size(); ++i) {
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(indicesDesc[i]));
  }
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(valueDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(inputDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(outputDesc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_put,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::IndexPutKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          int,
                          int64_t) {}
