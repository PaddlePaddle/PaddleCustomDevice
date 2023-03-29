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

namespace custom_kernel {

static inline int64_t ComputeAxis(int64_t axis, int64_t rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto axis = axis_scalar.to<int>();
  auto ins_size = ins.size();

  PADDLE_ENFORCE_NOT_NULL(
      ins[0],
      phi::errors::NotFound("The first input tensor is not initalized."));

  axis = ComputeAxis(static_cast<int64_t>(axis),
                     static_cast<int64_t>(ins[0]->dims().size()));

  // mlu should do sth
  // init ins tensors
  std::vector<const void*> inputs;
  std::vector<MLUCnnlTensorDesc> input_descs;
  std::vector<cnnlTensorDescriptor_t> desc_vector;
  for (size_t i = 0; i < ins_size; i++) {
    input_descs.emplace_back(MLUCnnlTensorDesc(
        *ins[i], CNNL_LAYOUT_ARRAY, ToCnnlDataType(ins[i]->dtype())));
    desc_vector.push_back(input_descs.back().get());
    inputs.push_back(GetBasePtr(ins[i]));
  }
  // init out tensors
  MLUCnnlTensorDesc output_desc(
      *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));

  // MLU should do sth
  MLUCnnl::Concat(dev_ctx,
                  ins_size,
                  axis,
                  desc_vector.data(),
                  inputs.data(),
                  output_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const phi::DenseTensor*>& ins,
                      const phi::DenseTensor& dout,
                      const phi::Scalar& axis_scalar,
                      std::vector<phi::DenseTensor*> outs) {
  auto axis = axis_scalar.to<int>();
  int split_num = ins.size();

  PADDLE_ENFORCE_NOT_NULL(
      ins[0],
      phi::errors::NotFound("The first input tensor is not initalized."));

  axis = ComputeAxis(static_cast<int64_t>(axis),
                     static_cast<int64_t>(ins[0]->dims().size()));
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument("concat_grad: axis should be larger than or "
                                   "equal to 0, but received axis is %d.",
                                   axis));
  PADDLE_ENFORCE_LT(axis,
                    dout.dims().size(),
                    phi::errors::InvalidArgument(
                        "concat_grad: axis should be less than ins[0]->dims()!"
                        "But received axis is %d, while ins[0]->dims()"
                        "size is %d.",
                        axis,
                        dout.dims().size()));
  // get output tensor that the name is not kEmptyVarName
  std::vector<void*> outputs_vec;
  std::vector<Tensor> tmp_outputs_vec;
  std::vector<MLUCnnlTensorDesc> output_descs;
  std::vector<cnnlTensorDescriptor_t> descs_vec;
  for (size_t j = 0; j < outs.size(); ++j) {
    if (outs[j] && outs[j]->numel() != 0UL) {
      dev_ctx.template Alloc<T>(outs[j]);
      output_descs.emplace_back(MLUCnnlTensorDesc(*outs[j]));
      outputs_vec.push_back(GetBasePtr(outs[j]));
    } else {
      Tensor tmp_tensor;
      tmp_tensor.Resize(ins[j]->dims());
      dev_ctx.template Alloc<T>(&tmp_tensor);
      tmp_outputs_vec.push_back(tmp_tensor);
      output_descs.emplace_back(MLUCnnlTensorDesc(*ins[j]));
      outputs_vec.push_back(GetBasePtr(&(tmp_outputs_vec.back())));
    }
    descs_vec.push_back(output_descs.back().get());
  }

  MLUCnnlTensorDesc out_grad_desc(dout);
  MLUCnnl::Split(dev_ctx,
                 static_cast<int>(split_num),
                 static_cast<int>(axis),
                 out_grad_desc.get(),
                 GetBasePtr(&dout),
                 descs_vec.data(),
                 outputs_vec.data());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(concat,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatKernel,
                          uint8_t,
                          int,
                          int64_t,
                          bool,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(concat_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatGradKernel,
                          uint8_t,
                          int,
                          int64_t,
                          bool,
                          float,
                          phi::dtype::float16) {}
