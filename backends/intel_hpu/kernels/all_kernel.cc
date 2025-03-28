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
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class All : public HpuOperator {
 public:
  All() : HpuOperator("all_fwd_") {}

  void AddNode(ConvertTensors& ct, ns_Reduction::ParamsV2& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::string guid_constant = "constant_" + SynDataTypeToStr(inputs[0].type);
    std::string guid_not_equal =
        "not_equal_fwd_" + SynDataTypeToStr(inputs[0].type);
    std::string guid_f32_cast = "cast_i8_to_f32";
    std::string guid_reduce = "reduce_prod_multi_dim_fwd_f32";
    std::string guid_f32_i8_cast = "cast_f32_to_i8";

    std::vector<synTensor> syn_inputs;
    synStatus status = synFail;

    if (inputs[0].type == syn_type_int8) {
      for (size_t i = 0; i < inputs.size(); i++) {
        syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                          inputs[i].type,
                                          inputs[i].dims,
                                          true,
                                          inputs[i].name));
      }
    } else {
      // not equal with zero
      for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<synTensor> zero_out;
        auto zero_tensor = createTensor(inputs[i].dims.size(),
                                        inputs[i].type,
                                        inputs[i].dims,
                                        false,
                                        "zero_tensor");
        zero_out.push_back(zero_tensor);

        ns_ConstantKernel::Params zeroParams;
        zeroParams.constant.i = 0;

        status = synNodeCreate(graphHandle_,
                               nullptr,
                               zero_out.data(),
                               0,
                               zero_out.size(),
                               &zeroParams,
                               sizeof(zeroParams),
                               guid_constant.c_str(),
                               "zero_constant",
                               nullptr,
                               nullptr);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synNodeCreate (any/full) failed = ",
                 status);

        std::vector<synTensor> equal_inputs;
        equal_inputs.push_back(createTensor(inputs[i].dims.size(),
                                            inputs[i].type,
                                            inputs[i].dims,
                                            true,
                                            inputs[i].name));
        equal_inputs.push_back(zero_tensor);

        std::vector<synTensor> equal_outputs;
        auto equal = createTensor(inputs[i].dims.size(),
                                  syn_type_int8,
                                  inputs[i].dims,
                                  false,
                                  "equal");
        equal_outputs.push_back(equal);

        status = synNodeCreate(graphHandle_,
                               equal_inputs.data(),
                               equal_outputs.data(),
                               equal_inputs.size(),
                               equal_outputs.size(),
                               nullptr,
                               0,
                               guid_not_equal.c_str(),
                               "not_equal",
                               nullptr,
                               nullptr);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synNodeCreate (any/not_equal) failed = ",
                 status);

        syn_inputs.push_back(equal);
      }
    }
    // Cast to f32
    std::vector<synTensor> f32_outputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      f32_outputs.push_back(createTensor(inputs[i].dims.size(),
                                         syn_type_float,
                                         inputs[i].dims,
                                         false,
                                         "f32_outputs"));
    }

    status = synNodeCreate(graphHandle_,
                           syn_inputs.data(),
                           f32_outputs.data(),
                           syn_inputs.size(),
                           f32_outputs.size(),
                           nullptr,
                           0,
                           guid_f32_cast.c_str(),
                           "all_cast_f32",
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    // Reduce
    std::vector<synTensor> reduce_outputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      reduce_outputs.push_back(createTensor(outputs[i].dims.size(),
                                            syn_type_float,
                                            outputs[i].dims,
                                            false,
                                            "reduce_outputs"));
    }

    status = synNodeCreate(graphHandle_,
                           f32_outputs.data(),
                           reduce_outputs.data(),
                           f32_outputs.size(),
                           reduce_outputs.size(),
                           &params,
                           sizeof(params),
                           guid_reduce.c_str(),
                           "all_prod_reduce",
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    // Cast to i8
    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }

    status = synNodeCreate(graphHandle_,
                           reduce_outputs.data(),
                           syn_outputs.data(),
                           reduce_outputs.size(),
                           syn_outputs.size(),
                           nullptr,
                           0,
                           guid_f32_i8_cast.c_str(),
                           "all_cast_i8",
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void AllKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  if (x.numel() == 0) {
    return;
  }

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  auto rank = static_cast<int32_t>(x.dims().size());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  ns_Reduction::ParamsV2 params = {};
  OpCacheOperator op_info;
  params.keepDim = keep_dim;
  if (dims.size() != 0) {
    for (size_t i = 0; i < dims.size(); i++) {
      auto dim = CanonicalAxis(static_cast<int64_t>(dims[i]),
                               static_cast<int64_t>(rank));
      params.reductionDimensionMask |= 1 << (rank - dim - 1);
    }
  } else {
    params.reductionDimensionMask = (1 << rank) - 1;
  }

  op_info.prepareOpInfo<T, ns_Reduction::ParamsV2>(
      "AllKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    All op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }
  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(all,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::AllKernel,
                          bool,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int32_t) {}
