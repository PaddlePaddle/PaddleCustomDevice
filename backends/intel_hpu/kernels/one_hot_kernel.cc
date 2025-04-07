// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class OneHotOperator : public HpuOperator {
 public:
  OneHotOperator(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}
  void AddNode(ConvertTensors& ct, ns_OneHotKernel::Params params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);
    std::vector<synTensor> syn_inputs;
    if (inputs[0].type == syn_type_int32) {
      for (size_t i = 0; i < inputs.size(); i++) {
        syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                          inputs[i].type,
                                          inputs[i].dims,
                                          true,
                                          inputs[i].name));
      }
    } else {
      for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<synTensor> x_i64;
        x_i64.push_back(createTensor(inputs[i].dims.size(),
                                     inputs[i].type,
                                     inputs[i].dims,
                                     true,
                                     inputs[i].name));
        std::vector<synTensor> x_i32;
        auto x_cast = createTensor(inputs[i].dims.size(),
                                   syn_type_int32,
                                   inputs[i].dims,
                                   false,
                                   "x_cast");
        x_i32.push_back(x_cast);

        std::string guid_cast = "cast_i64_to_i32";
        synStatus status = synNodeCreate(graphHandle_,
                                         x_i64.data(),
                                         x_i32.data(),
                                         x_i64.size(),
                                         x_i32.size(),
                                         nullptr,
                                         0,
                                         guid_cast.c_str(),
                                         "cast_x",
                                         nullptr,
                                         nullptr);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synNodeCreate cast_x failed = ",
                 status);
        syn_inputs.push_back(x_cast);
      }
    }

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     inputs.size(),
                                     outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     pName_.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
  std::string pName_;
};

template <typename T, typename Context>
void OneHotRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& num_classes,
                     phi::DataType dtype,
                     bool allow_out_of_range,
                     phi::DenseTensor* out) {
  // allocate memory on device.
  ConvertTensors ct;
  ct.Add(x);
  dev_ctx.template Alloc<float>(out);
  int depth = num_classes.to<int>();
  auto out_dims = out->dims();
  out_dims[out_dims.size() - 1] = depth;
  out->Resize(out_dims);
  ct.Add(out, false);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  ns_OneHotKernel::Params params{-1, depth, 1, 0};

  OpCacheOperator op_info;
  op_info.prepareOpInfo<float, ns_OneHotKernel::Params>(
      "one_hot_fwd", {inputs_dims}, &params);

  auto recipe = op_info.GetRecipe();
  if (recipe == nullptr) {
    // compile
    OneHotOperator op(op_info.guid_, "one_hot_op");
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& num_classes_s,
                  phi::DenseTensor* out) {
  custom_kernel::OneHotRawKernel<T, Context>(
      dev_ctx, x, num_classes_s, phi::DataType::FLOAT32, false, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(one_hot_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::OneHotRawKernel,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(one_hot,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::OneHotKernel,
                          int32_t,
                          int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
