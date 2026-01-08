// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs.h"
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {
struct StridedCopyParams {
  bool is_src_contiguous;
  synStridedOpParams view_params;
  synStridedOpParams insert_params;
  std::vector<int64_t> dims;
};

class StridedCopy : public HpuFusedOperator {
 public:
  StridedCopy() : HpuFusedOperator("strided_insert") {}

  void AddNode(ConvertTensors& ct, StridedCopyParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    synSectionHandle section = createSection();

    synTensor dst = createTensorFromCT(&ct, 0, true, section);
    synTensor src = createTensorFromCT(&ct, 1);

    std::vector<synTensor> insert_inputs;
    insert_inputs.push_back(dst);

    auto out = createTensorFromCT(&ct, 0, false, section);

    if (!params.is_src_contiguous) {
      std::string guid_view = "strided_view";

      auto src_contiguous =
          createTensorNoPresist("src_contiguous", inputs[1].type, params.dims);

      std::vector<synTensor> view_inputs;
      view_inputs.push_back(src);
      std::vector<synTensor> view_outputs;
      view_outputs.push_back(src_contiguous);

      synStatus status = synNodeCreate(graphHandle_,
                                       view_inputs.data(),
                                       view_outputs.data(),
                                       view_inputs.size(),
                                       view_outputs.size(),
                                       &params.view_params,
                                       sizeof(params.view_params),
                                       guid_view.c_str(),
                                       "Strided View",
                                       nullptr,
                                       nullptr);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synNodeCreate () failed = %d",
               status);

      insert_inputs.push_back(src_contiguous);
    } else {
      insert_inputs.push_back(src);
    }

    std::vector<synTensor> insert_outputs;
    insert_outputs.push_back(out);

    synStatus status = synNodeCreate(graphHandle_,
                                     insert_inputs.data(),
                                     insert_outputs.data(),
                                     insert_inputs.size(),
                                     insert_outputs.size(),
                                     &params.insert_params,
                                     sizeof(params.insert_params),
                                     guid_.c_str(),
                                     "Strided Copy",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       phi::DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  phi::DenseTensor flat_input(input);

  StridedCopyParams params;
  params.is_src_contiguous = input.meta().is_contiguous();
  params.dims = phi::vectorize<int64_t>(input.dims());

  if (!params.is_src_contiguous) {
    params.view_params.baseOffset = meta.offset / sizeof(T);
    std::vector<int32_t> input_strides = phi::vectorize<int32_t>(meta.strides);
    auto rank = input_strides.size();
    for (size_t i = 0; i < rank; i++) {
      params.view_params.strides[rank - 1 - i] = input_strides[i];
    }
    for (size_t i = rank; i < HABANA_DIM_MAX; i++) {
      params.view_params.strides[i] = 0;
    }
    // calculate inputs dim
    std::vector<int64_t> input_dims = phi::vectorize<int64_t>(meta.dims);
    uint64_t lastElementOffset = 0;
    for (size_t i = 0; i < rank; i++) {
      lastElementOffset += input_strides[i] * (input_dims[i] - 1);
    }
    int64_t numOfInputElements =
        params.view_params.baseOffset + lastElementOffset + 1;

    phi::DenseTensorMeta fake_meta({input.dtype(), {numOfInputElements}});
    flat_input.set_meta(fake_meta);
  }

  uint64_t base_offset = offset / sizeof(T);
  params.insert_params.baseOffset = base_offset;
  auto rank = out_stride.size();
  for (size_t i = 0; i < rank; i++) {
    params.insert_params.strides[rank - 1 - i] = out_stride[i];
  }

  for (size_t i = rank; i < HABANA_DIM_MAX; i++) {
    params.insert_params.strides[i] = 0;
  }

  meta = out->meta();

  for (int64_t i = meta.strides.size() - 2; i >= 0; i--) {
    if (meta.strides[i] != meta.strides[i + 1] * meta.dims[i + 1]) {
      meta.dims[i + 1] = meta.strides[i] / meta.strides[i + 1];
    }
  }

  int64_t total_size = 1;
  for (int i = 0; i < meta.dims.size(); i++) {
    total_size *= meta.dims[i];
  }

  int64_t expand_dim = base_offset / total_size;

  if (expand_dim > 0) {
    std::vector<int64_t> new_dims(meta.dims.size() + 1);
    new_dims[0] = expand_dim + 1;
    for (int i = 0; i < meta.dims.size(); ++i) {
      new_dims[i + 1] = meta.dims[i];
    }
    meta.dims = common::make_ddim(new_dims);
  }

  meta.strides = common::make_ddim(out_stride);
  meta.offset = 0;
  out->set_meta(meta);

  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(out);

  if (params.is_src_contiguous)
    ct.Add(input);
  else
    ct.Add(flat_input);
  ct.Add(out, false);

  std::vector<DIMS> outputs_dims = ct.GetDims(false);
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, StridedCopyParams>(
      "StridedCopyKernel", outputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    StridedCopy op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);

  meta.offset = offset;
  meta.dims = common::make_ddim(dims);
  out->set_meta(meta);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(strided_copy,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::StridedCopyKernel,
                          uint8_t,
                          int8_t,
                          bool,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::float8_e4m3fn) {}
