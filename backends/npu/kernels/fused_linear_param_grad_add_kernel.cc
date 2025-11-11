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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void FusedLinearParamGradAdd(const Context &dev_ctx,
                             const phi::DenseTensor &x,
                             const phi::DenseTensor &dout,
                             const paddle::optional<phi::DenseTensor> &dweight,
                             const paddle::optional<phi::DenseTensor> &dbias,
                             bool multi_precision,
                             bool has_bias,
                             phi::DenseTensor *dweight_out,
                             phi::DenseTensor *dbias_out) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  int64_t K = x.dims()[x.dims().size() - 1];
  int64_t N = dout.dims()[dout.dims().size() - 1];

  phi::DenseTensor reshape_x(x);
  reshape_x.Resize({x.numel() / K, K});
  phi::DenseTensor reshape_dout(dout);
  reshape_dout.Resize({dout.numel() / N, N});

  if (dweight_out && dweight) {
    *dweight_out = dweight.get();
    if (multi_precision) {
      PADDLE_ENFORCE_EQ(
          dweight_out->dtype(),
          phi::CppTypeToDataType<MT>::Type(),
          phi::errors::InvalidArgument("Invaid data type error."));
    } else {
      PADDLE_ENFORCE_EQ(
          dweight_out->dtype(),
          phi::CppTypeToDataType<T>::Type(),
          phi::errors::InvalidArgument("Invaid data type error."));
    }
  } else {
    dweight_out->Resize(phi::make_ddim({K, N}));
    if (multi_precision) {
      dev_ctx.template Alloc<MT>(dweight_out);
    } else {
      dev_ctx.template Alloc<T>(dweight_out);
    }
  }

  if (has_bias && dbias_out) {
    dev_ctx.template Alloc<T>(dbias_out);
  }

  float alpha = 1.0;
  float beta = 1.0;

  phi::DenseTensor new_dweight;
  if (dweight) {
    new_dweight = dweight.get();
  } else {
    phi::DenseTensorMeta dweight_meta = {x.dtype(), {K, N}};
    new_dweight.set_meta(dweight_meta);
    FillNpuTensorWithConstant<T>(&new_dweight, dev_ctx, static_cast<T>(0));
    new_dweight.Resize({K, N});
  }

  int64_t trans_a = 1;
  int64_t trans_b = 0;
  int8_t cube_math_type = 0;
  bool keep_dim = false;
  EXEC_NPU_CMD(aclnnGemm,
               dev_ctx,
               reshape_x,
               reshape_dout,
               new_dweight,
               alpha,
               beta,
               trans_a,
               trans_b,
               *dweight_out,
               cube_math_type);
  if (has_bias) {
    phi::IntArray axis = {0};

    phi::DenseTensor new_dbias;
    if (dbias) {
      new_dbias = dbias.get();
    } else {
      phi::DenseTensorMeta new_dbias_meta = {x.dtype(), {N}};
      new_dbias.set_meta(new_dbias_meta);
      FillNpuTensorWithConstant<T>(&new_dbias, dev_ctx, static_cast<T>(0));
    }

    auto dst_dtype = ConvertToNpuDtype(reshape_dout.dtype());

    phi::DenseTensor bias_sum;
    phi::DenseTensorMeta bias_sum_meta = {x.dtype(), {N}};
    bias_sum.set_meta(bias_sum_meta);
    dev_ctx.template Alloc<T>(&bias_sum);
    EXEC_NPU_CMD(aclnnReduceSum,
                 dev_ctx,
                 reshape_dout,
                 axis,
                 keep_dim,
                 dst_dtype,
                 bias_sum);
    phi::Scalar add_alpha = 1.0;
    EXEC_NPU_CMD(aclnnAdd, dev_ctx, bias_sum, new_dbias, add_alpha, *dbias_out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fused_linear_param_grad_add,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FusedLinearParamGradAdd,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
