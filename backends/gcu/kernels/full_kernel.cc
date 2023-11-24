// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "common/common.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/common/data_type.h"

namespace custom_kernel {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  auto shape_vec = shape.GetData();
  auto out_dim = phi::make_ddim(shape_vec);
  out->ResizeAndAllocate(out_dim);
  dev_ctx.template Alloc<T>(out);
  FillGcuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(val.to<T>()));
  out->Resize(out_dim);
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "full_like", full_like);
#define FULL_LIKE_CASE(out, DATA_TYPE, val)                       \
  case DATA_TYPE: {                                               \
    typedef typename ::phi::DataTypeToCppType<DATA_TYPE>::type K; \
    K value = val.to<K>();                                        \
    *out = full_like(dev_ctx, x, value);                          \
    break;                                                        \
  }

#define FULL_LIKE(out, dtype, val)                                             \
  switch (dtype) {                                                             \
    FULL_LIKE_CASE(out, phi::DataType::INT8, val)                              \
    FULL_LIKE_CASE(out, phi::DataType::INT16, val)                             \
    FULL_LIKE_CASE(out, phi::DataType::INT32, val)                             \
    FULL_LIKE_CASE(out, phi::DataType::INT64, val)                             \
    FULL_LIKE_CASE(out, phi::DataType::UINT8, val)                             \
    FULL_LIKE_CASE(out, phi::DataType::FLOAT16, val)                           \
    FULL_LIKE_CASE(out, phi::DataType::BFLOAT16, val)                          \
    FULL_LIKE_CASE(out, phi::DataType::FLOAT32, val)                           \
    FULL_LIKE_CASE(out, phi::DataType::FLOAT64, val)                           \
    FULL_LIKE_CASE(out, phi::DataType::BOOL, val)                              \
    default: {                                                                 \
      PADDLE_ENFORCE(                                                          \
          false,                                                               \
          phi::errors::InvalidArgument("Invalid scalar type %s",               \
                                       phi::DataTypeToString(dtype).c_str())); \
    }                                                                          \
  }
    FULL_LIKE(out, dtype, val);
    PADDLE_GCU_KERNEL_END("full_like", full_like);
  } else {
    dev_ctx.template Alloc<T>(out);

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    float value = val.to<float>();

    GcuAttributeMap attrs;
    attrs["dtype"] = static_cast<int>(dtype);
    attrs["value"] = value;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "fill_any_like",
              dev_ctx);
  }
}  // namespace custom_kernel

template <typename T, typename Context>
void FullBatchSizeLikeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const std::vector<int>& shape,
                             const phi::Scalar& val,
                             phi::DataType dtype,
                             int x_batch_size_dim,
                             int out_batch_size_dim,
                             phi::DenseTensor* out) {
  if (x.lod().size() && x_batch_size_dim == 0) {
    // set the correct batch size for the LoDTensor.
    auto odims = out->dims();
    odims[out_batch_size_dim] = static_cast<int>(x.lod().back().size()) - 1;
    custom_kernel::FullKernel<T, Context>(
        dev_ctx, phi::vectorize(odims), val, dtype, out);
  }
  custom_kernel::FullLikeKernel<T, Context>(dev_ctx, x, val, dtype, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(full,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FullKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(full_like,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FullLikeKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

// PD_REGISTER_PLUGIN_KERNEL(full_batch_size_like,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::FullBatchSizeLikeKernel,
//                           int,
//                           float,
//                           phi::dtype::float16) {
//   kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
// }
