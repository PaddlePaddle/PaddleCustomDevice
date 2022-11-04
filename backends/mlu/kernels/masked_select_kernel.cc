// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void MaskedSelectKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& mask,
                        phi::DenseTensor* out) {
  auto input_dim = x.dims();
  auto mask_dim = mask.dims();
  PADDLE_ENFORCE_EQ(input_dim,
                    mask_dim,
                    phi::errors::InvalidArgument(
                        "The dim size of input and mask in OP(masked_selected) "
                        "must be equal, but got input dim:(%ld), mask dim: "
                        "(%ld). Please check input "
                        "value.",
                        input_dim,
                        mask_dim));

  Tensor number;
  number.Resize({1});
  void* number_ptr = dev_ctx.template Alloc<int32_t>(&number);
  out->Resize(mask.dims());
  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc mask_desc(mask);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::Mask(dev_ctx,
                CNNL_MASKED_SELECT,
                input_desc.get(),
                GetBasePtr(&x),
                mask_desc.get(),
                GetBasePtr(&mask),
                nullptr, /* value_desc */
                nullptr, /* value */
                nullptr, /* scale */
                out_desc.get(),
                GetBasePtr(out),
                static_cast<uint32_t*>(number_ptr));
}

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& mask,
                            const phi::DenseTensor& out_grad,
                            phi::DenseTensor* x_grad) {
  C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());
  Tensor mask_int32, out_size;
  std::vector<int32_t> out_size_vec;
  mask_int32.Resize({mask.dims()});
  dev_ctx.template Alloc<int32_t>(&mask_int32);
  out_size.Resize({1});
  dev_ctx.template Alloc<int32_t>(&out_size);

  MLUCnnlTensorDesc mask_desc(mask);
  MLUCnnlTensorDesc mask_int32_desc(mask_int32);
  MLUCnnlTensorDesc out_size_desc(out_size);
  auto cast_type = GetCastDataType(mask.dtype(), DataType::INT32);
  MLUCnnl::Cast(dev_ctx,
                cast_type,
                mask_desc.get(),
                GetBasePtr(&mask),
                mask_int32_desc.get(),
                GetBasePtr(&mask_int32));

  auto mask_int32_dim = phi::vectorize(mask_int32.dims());
  std::vector<int32_t> reduce_dims;
  for (size_t i = 0; i < mask_int32_dim.size(); i++) {
    reduce_dims.push_back(static_cast<int>(i));
  }

  std::string reduce_name = "reduce_sum";
  cnnlReduceOp_t reduce_op = GetMLUCnnlReduceOp(reduce_name);
  MLUCnnlReduceDesc reduce_desc(reduce_dims,
                                reduce_op,
                                ToCnnlDataType<int32_t>(),
                                CNNL_NOT_PROPAGATE_NAN,
                                CNNL_REDUCE_NO_INDICES,
                                CNNL_32BIT_INDICES);

  MLUCnnl::Reduce(dev_ctx,
                  true,
                  reduce_desc.get(),
                  nullptr,
                  mask_int32_desc.get(),
                  GetBasePtr(&mask_int32),
                  0,
                  nullptr,
                  nullptr,
                  out_size_desc.get(),
                  GetBasePtr(&out_size));

  TensorToVector(dev_ctx, out_size, dev_ctx, &out_size_vec);
  dev_ctx.Wait();

  Tensor mask_int32_tmp;
  mask_int32_tmp = mask_int32;
  mask_int32_tmp.Resize({mask_int32.numel()});
  Tensor topk_v2_out;
  Tensor indices_int32;
  topk_v2_out.Resize({mask_int32.numel()});
  indices_int32.Resize({mask_int32.numel()});
  dev_ctx.template Alloc<int32_t>(&topk_v2_out);
  dev_ctx.template Alloc<int32_t>(&indices_int32);

  MLUCnnlTensorDesc topk_v2_out_desc(topk_v2_out);
  MLUCnnlTensorDesc indices_int32_desc(indices_int32);
  MLUCnnlTensorDesc mask_int32_tmp_desc(mask_int32_tmp);

  const int dim = 0;
  MLUCnnl::TopK(dev_ctx,
                mask_int32.numel(),
                dim,
                true,
                false,
                mask_int32_tmp_desc.get(),
                GetBasePtr(&mask_int32_tmp),
                topk_v2_out_desc.get(),
                GetBasePtr(&topk_v2_out),
                indices_int32_desc.get(),
                GetBasePtr(&indices_int32));

  Tensor indices_int32_out;
  indices_int32_out.Resize({out_size_vec[0]});
  dev_ctx.template Alloc<int32_t>(&indices_int32_out);
  AsyncMemCpyD2D(nullptr,
                 stream,
                 GetBasePtr(&indices_int32_out),
                 GetBasePtr(&indices_int32),
                 out_size_vec[0] * sizeof(int32_t));
  dev_ctx.Wait();

  Tensor out_grad_tmp_out;
  out_grad_tmp_out.Resize({out_size_vec[0]});
  dev_ctx.template Alloc<T>(&out_grad_tmp_out);
  MLUCnnlTensorDesc out_grad_tmp_out_desc(out_grad_tmp_out);
  AsyncMemCpyD2D(nullptr,
                 stream,
                 GetBasePtr(&out_grad_tmp_out),
                 GetBasePtr(&out_grad),
                 out_size_vec[0] * sizeof(T));
  dev_ctx.Wait();

  Tensor indices_int32_tmp;
  indices_int32_tmp = indices_int32_out;
  indices_int32_tmp.Resize({out_size_vec[0], 1});
  MLUCnnlTensorDesc indices_int32_tmp_desc(indices_int32_tmp);

  const cnnlScatterNdMode_t mode = CNNL_SCATTERND_UPDATE;
  x_grad->Resize({x_grad->numel()});
  dev_ctx.template Alloc<T>(x_grad);
  MLUCnnlTensorDesc x_grad_desc(*x_grad);
  MLUCnnl::ScatterNd(dev_ctx,
                     mode,
                     indices_int32_tmp_desc.get(),
                     GetBasePtr(&indices_int32_tmp),
                     out_grad_tmp_out_desc.get(),
                     GetBasePtr(&out_grad_tmp_out),
                     nullptr,
                     nullptr,
                     x_grad_desc.get(),
                     GetBasePtr(x_grad));
  x_grad->Resize(mask.dims());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(masked_select,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MaskedSelectKernel,
                          phi::dtype::float16,
                          float,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(masked_select_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MaskedSelectGradKernel,
                          phi::dtype::float16,
                          float,
                          int) {}
