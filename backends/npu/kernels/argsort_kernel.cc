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
#include "paddle/phi/backends/custom/custom_context.h"

namespace custom_kernel {

template <typename T>
static void TranposeNPU(const phi::CustomContext& dev_ctx,
                        const aclrtStream& stream,
                        std::vector<int64_t>* perm,
                        const phi::DenseTensor& in,
                        phi::DenseTensor* out) {
  dev_ctx.Alloc<T>(out);
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(in)
      .AddInput(dev_ctx, std::move(*perm))
      .AddOutput(*out)
      .Run(stream);
}

static void CastToInt64(const phi::CustomContext& dev_ctx,
                        const aclrtStream& stream,
                        const phi::DenseTensor& in,
                        phi::DenseTensor* out) {
  dev_ctx.Alloc<int64_t>(out);
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_INT64)
      .Run(stream);
}

static void CastToFP32(const phi::CustomContext& dev_ctx,
                       const aclrtStream& stream,
                       const phi::DenseTensor& in,
                       phi::DenseTensor* out) {
  dev_ctx.Alloc<float>(out);
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_FLOAT)
      .Run(stream);
}

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const phi::DenseTensor& in,
                   int axis,
                   bool descending,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {
  // TODO(Aganlengzi): Sort may change the input data !
  // Here we make a deepcopy to workaround before it is fixed.
  if (in.dims().size() == 0) {
    dev_ctx.template Alloc<T>(output);
    TensorCopy(dev_ctx, in, false, output);
    dev_ctx.template Alloc<int64_t>(indices);
    FillNpuTensorWithConstant<int64_t>(
        indices, dev_ctx, static_cast<int64_t>(0));
    indices->Resize(phi::make_ddim({}));
    return;
  }

  phi::DenseTensor input;
  TensorCopy(dev_ctx, in, false, &input);

  auto in_dims = input.dims();
  axis = (axis < 0) ? (in_dims.size() + axis) : axis;

  auto stream = dev_ctx.stream();

  phi::DenseTensor indices_tmp;
  phi::DenseTensorMeta indices_tmp_meta = {phi::DataType::INT32,
                                           indices->dims()};
  indices_tmp.set_meta(indices_tmp_meta);

  if (input.dtype() == phi::DataType::INT64) {
    phi::DenseTensor input_fp32;
    phi::DenseTensorMeta input_fp32_meta = {phi::DataType::FLOAT32,
                                            input.dims()};
    input_fp32.set_meta(input_fp32_meta);
    CastToFP32(dev_ctx, stream, input, &input_fp32);

    phi::DenseTensor output_fp32;
    phi::DenseTensorMeta output_fp32_meta = {phi::DataType::FLOAT32,
                                             output->dims()};
    output_fp32.set_meta(output_fp32_meta);

    if (axis == -1 || axis + 1 == in_dims.size()) {
      dev_ctx.template Alloc<float>(&output_fp32);
      dev_ctx.template Alloc<int32_t>(&indices_tmp);
      NpuOpRunner runner;
      runner.SetType("Sort")
          .AddInput(input_fp32)
          .AddOutput(output_fp32)
          .AddOutput(indices_tmp)
          .AddAttr("axis", -1)
          .AddAttr("descending", descending)
          .Run(stream);
      CastToInt64(dev_ctx, stream, output_fp32, output);
    } else {
      std::vector<int64_t> perm;
      for (int64_t i = 0; i < in_dims.size(); i++) {
        perm.emplace_back(i);
      }
      std::swap(perm[axis], perm[in_dims.size() - 1]);

      std::vector<int64_t> shape;
      for (size_t i = 0; i < perm.size(); i++) {
        shape.emplace_back(in_dims[perm[i]]);
      }
      auto trans_dims = phi::make_ddim(shape);

      phi::DenseTensor trans_input;
      phi::DenseTensorMeta trans_input_meta = {input_fp32.dtype(), trans_dims};
      trans_input.set_meta(trans_input_meta);
      TranposeNPU<float>(dev_ctx, stream, &perm, input_fp32, &trans_input);

      phi::DenseTensor trans_output;
      phi::DenseTensorMeta trans_output_meta = {input_fp32.dtype(), trans_dims};
      trans_output.set_meta(trans_output_meta);
      dev_ctx.template Alloc<float>(&trans_output);
      phi::DenseTensor trans_indices;
      phi::DenseTensorMeta trans_indices_meta = {phi::DataType::INT32,
                                                 trans_dims};
      trans_indices.set_meta(trans_indices_meta);
      dev_ctx.template Alloc<int32_t>(&trans_indices);

      NpuOpRunner runner;
      runner.SetType("Sort")
          .AddInput(trans_input)
          .AddOutput(trans_output)
          .AddOutput(trans_indices)
          .AddAttr("axis", -1)
          .AddAttr("descending", descending)
          .Run(stream);

      TranposeNPU<float>(dev_ctx, stream, &perm, trans_output, &output_fp32);
      TranposeNPU<int32_t>(dev_ctx, stream, &perm, trans_indices, &indices_tmp);

      CastToInt64(dev_ctx, stream, output_fp32, output);
    }
  } else {
    if (axis == -1 || axis + 1 == in_dims.size()) {
      dev_ctx.template Alloc<T>(output);
      dev_ctx.template Alloc<int32_t>(&indices_tmp);

      NpuOpRunner runner;
      runner.SetType("Sort")
          .AddInput(input)
          .AddOutput(*output)
          .AddOutput(indices_tmp)
          .AddAttr("axis", -1)
          .AddAttr("descending", descending)
          .Run(stream);
    } else {
      std::vector<int64_t> perm;
      for (int64_t i = 0; i < in_dims.size(); i++) {
        perm.emplace_back(i);
      }
      std::swap(perm[axis], perm[in_dims.size() - 1]);

      std::vector<int64_t> shape;
      for (size_t i = 0; i < perm.size(); i++) {
        shape.emplace_back(in_dims[perm[i]]);
      }
      auto trans_dims = phi::make_ddim(shape);

      phi::DenseTensor trans_input;
      phi::DenseTensorMeta trans_input_meta = {input.dtype(), trans_dims};
      trans_input.set_meta(trans_input_meta);
      TranposeNPU<T>(dev_ctx, stream, &perm, input, &trans_input);

      phi::DenseTensor trans_output;
      phi::DenseTensorMeta trans_output_meta = {input.dtype(), trans_dims};
      trans_output.set_meta(trans_output_meta);
      dev_ctx.template Alloc<T>(&trans_output);

      phi::DenseTensor trans_indices;
      phi::DenseTensorMeta trans_indices_meta = {phi::DataType::INT32,
                                                 trans_dims};
      trans_indices.set_meta(trans_indices_meta);
      dev_ctx.template Alloc<int32_t>(&trans_indices);

      NpuOpRunner runner;
      runner.SetType("Sort")
          .AddInput(trans_input)
          .AddOutput(trans_output)
          .AddOutput(trans_indices)
          .AddAttr("axis", -1)
          .AddAttr("descending", descending)
          .Run(stream);

      TranposeNPU<T>(dev_ctx, stream, &perm, trans_output, output);
      TranposeNPU<int32_t>(dev_ctx, stream, &perm, trans_indices, &indices_tmp);
    }
  }

  CastToInt64(dev_ctx, stream, indices_tmp, indices);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argsort,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ArgsortKernel,
                          float,
                          int64_t,
                          phi::dtype::float16) {}
