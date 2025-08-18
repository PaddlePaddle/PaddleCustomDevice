// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class HpuFusedOperator : public HpuOperator {
 public:
  explicit HpuFusedOperator(const std::string& guid, bool is_eager = true)
      : HpuOperator(guid, is_eager) {}

  template <typename T>
  std::string guid_dtype() {
    if (std::is_same<T, phi::dtype::float16>::value) {
      return "f16";
    } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
      return "bf16";
    } else if (std::is_same<T, float>::value) {
      return "f32";
    } else if (std::is_same<T, phi::dtype::float8_e4m3fn>::value) {
      return "hf8";
    } else if (std::is_same<T, int16_t>::value) {
      return "i16";
    } else if (std::is_same<T, int32_t>::value) {
      return "i32";
    } else if (std::is_same<T, bool>::value) {
      return "i8";
    } else if (std::is_same<T, int8_t>::value) {
      return "i8";
    } else if (std::is_same<T, int64_t>::value) {
      return "i64";
    } else {
      PD_CHECK(
          false, "[RUNTIME] synDataType not supported = %s", typeid(T).name());
    }
  }

  inline synTensor createTensorFromCT(ConvertTensors* ct,
                                      int idx,
                                      bool is_input = true,
                                      synSectionHandle section = nullptr) {
    PD_CHECK(ct != nullptr, "[RUNTIME] input ct is a nullptr");
    auto tensors = ct->GetTensors(is_input);
    synTensor t = createTensor(tensors[idx].dims.size(),
                               tensors[idx].type,
                               tensors[idx].dims,
                               true,
                               tensors[idx].name,
                               section);
    return t;
  }

  inline synTensor createTensorNoPresist(std::string name,
                                         synDataType dtype,
                                         std::vector<int64_t> dims,
                                         synSectionHandle section = nullptr) {
    synTensor t =
        createTensor(dims.size(), dtype, dims, false, name.c_str(), section);
    return t;
  }

  template <typename T>
  inline void AddNode_OP(std::vector<synTensor> outputs,
                         T params,
                         std::string guid,
                         std::string node_name) {
    synStatus status = synNodeCreate(graphHandle_,
                                     nullptr,
                                     outputs.data(),
                                     0,
                                     outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid.c_str(),
                                     node_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (",
             node_name,
             ") failed = ",
             status);
  }

  inline void AddNode_IO(std::vector<synTensor> inputs,
                         std::vector<synTensor> outputs,
                         std::string guid,
                         std::string node_name) {
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs.data(),
                                     inputs.size(),
                                     outputs.size(),
                                     nullptr,
                                     0,
                                     guid.c_str(),
                                     node_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (",
             node_name,
             ") failed = ",
             status);
  }

  template <typename T>
  inline void AddNode_IOP(std::vector<synTensor> inputs,
                          std::vector<synTensor> outputs,
                          T params,
                          std::string guid,
                          std::string node_name) {
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs.data(),
                                     inputs.size(),
                                     outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid.c_str(),
                                     node_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (",
             node_name,
             ") failed = ",
             status);
  }

  template <typename T>
  inline void AddNodeFull(std::vector<synTensor> outputs,
                          ns_ConstantKernel::Params params,
                          std::string node_name) {
    std::string guid = "constant_" + guid_dtype<T>();
    AddNode_OP<ns_ConstantKernel::Params>(outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeAdd(std::vector<synTensor> inputs,
                         std::vector<synTensor> outputs,
                         std::string node_name) {
    std::string guid = "add_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeSub(std::vector<synTensor> inputs,
                         std::vector<synTensor> outputs,
                         std::string node_name) {
    std::string guid = "sub_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeMultiply(std::vector<synTensor> inputs,
                              std::vector<synTensor> outputs,
                              std::string node_name) {
    std::string guid = "mult_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeDivide(std::vector<synTensor> inputs,
                            std::vector<synTensor> outputs,
                            std::string node_name) {
    std::string guid = "div_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeMaximum(std::vector<synTensor> inputs,
                             std::vector<synTensor> outputs,
                             std::string node_name) {
    std::string guid = "max_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeExp(std::vector<synTensor> inputs,
                         std::vector<synTensor> outputs,
                         std::string node_name) {
    std::string guid = "exp_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeAbs(std::vector<synTensor> inputs,
                         std::vector<synTensor> outputs,
                         std::string node_name) {
    std::string guid = "abs_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeSilu(std::vector<synTensor> inputs,
                          std::vector<synTensor> outputs,
                          std::string node_name) {
    std::string guid = "silu_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeLinear(std::vector<synTensor> inputs,
                            std::vector<synTensor> outputs,
                            std::string node_name) {
    std::string guid = "linear_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeLessEqual(std::vector<synTensor> inputs,
                               std::vector<synTensor> outputs,
                               std::string node_name) {
    std::string guid = "less_equal_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  inline void AddNodeReshape(std::vector<synTensor> inputs,
                             std::vector<synTensor> outputs,
                             std::string node_name) {
    AddNode_IO(inputs, outputs, "reshape", node_name);
  }

  template <typename T>
  inline void AddNodeWhere(std::vector<synTensor> inputs,
                           std::vector<synTensor> outputs,
                           std::string node_name) {
    std::string guid = "where_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  inline void AddNodeTranspose(std::vector<synTensor> inputs,
                               std::vector<synTensor> outputs,
                               synTransposeParams params,
                               std::string node_name) {
    AddNode_IOP<synTransposeParams>(
        inputs, outputs, params, "transpose", node_name);
  }

  void AddNodeCast(std::vector<synTensor> inputs,
                   std::vector<synTensor> outputs,
                   std::string guid,
                   std::string node_name) {
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  void AddNodeCast(std::vector<synTensor> inputs,
                   std::vector<synTensor> outputs,
                   ns_CastKernel::Params params,
                   std::string guid,
                   std::string node_name) {
    AddNode_IOP<ns_CastKernel::Params>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  void AddNodeConvertToFP8(std::vector<synTensor> inputs,
                           std::vector<synTensor> outputs,
                           ns_CastKernel::Params params,
                           std::string node_name) {
    std::string guid = "convert_to_fp8_" + guid_dtype<T>();
    AddNode_IOP<ns_CastKernel::Params>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  void AddNodeFP8Gemm(std::vector<synTensor> inputs,
                      std::vector<synTensor> outputs,
                      synGEMMParams params,
                      std::string node_name) {
    std::string guid = "fp8_gemm_" + guid_dtype<T>();
    AddNode_IOP<synGEMMParams>(inputs, outputs, params, guid, node_name);
  }

  inline void AddNodeGemm(std::vector<synTensor> inputs,
                          std::vector<synTensor> outputs,
                          synGEMMParams params,
                          std::string node_name) {
    AddNode_IOP<synGEMMParams>(inputs, outputs, params, "gemm", node_name);
  }

  inline void AddNodeBatchGemm(std::vector<synTensor> inputs,
                               std::vector<synTensor> outputs,
                               synGEMMParams params,
                               std::string node_name) {
    AddNode_IOP<synGEMMParams>(
        inputs, outputs, params, "batch_gemm", node_name);
  }

  template <typename T>
  inline void AddNodeCumsum(std::vector<synTensor> inputs,
                            std::vector<synTensor> outputs,
                            ns_CumSumKernel::Params params,
                            std::string node_name) {
    std::string guid = "cumsum_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_CumSumKernel::Params>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeIndexSelect(std::vector<synTensor> inputs,
                                 std::vector<synTensor> outputs,
                                 ns_GatherKernel::Params params,
                                 std::string node_name) {
    std::string guid = "gather_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_GatherKernel::Params>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeIndexSample(std::vector<synTensor> inputs,
                                 std::vector<synTensor> outputs,
                                 ns_GatherKernel::Params params,
                                 std::string node_name) {
    std::string guid = "gather_elements_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_GatherKernel::Params>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeIndexReduce(std::vector<synTensor> inputs,
                                 std::vector<synTensor> outputs,
                                 ns_IndexReduce::Params params,
                                 std::string node_name) {
    std::string guid = "index_reduce_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_IndexReduce::Params>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeReduceSum(std::vector<synTensor> inputs,
                               std::vector<synTensor> outputs,
                               ns_Reduction::Params params,
                               std::string node_name) {
    std::string guid = "reduce_sum_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_Reduction::Params>(inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeReduceMax(std::vector<synTensor> inputs,
                               std::vector<synTensor> outputs,
                               ns_Reduction::Params params,
                               std::string node_name) {
    std::string guid = "reduce_max_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_Reduction::Params>(inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeMaximumMultidimensional(std::vector<synTensor> inputs,
                                             std::vector<synTensor> outputs,
                                             ns_Reduction::ParamsV2 params,
                                             std::string node_name) {
    std::string guid = "reduce_max_multi_dim_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_Reduction::ParamsV2>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeScatterAdd(std::vector<synTensor> inputs,
                                std::vector<synTensor> outputs,
                                ns_ScatterKernel::Params params,
                                std::string node_name) {
    std::string guid = "unsorted_scatter_add_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_ScatterKernel::Params>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeScatter(std::vector<synTensor> inputs,
                             std::vector<synTensor> outputs,
                             std::string node_name) {
    std::string guid = "scatter_nd_onnx_fwd_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  inline void AddNodeConcat(std::vector<synTensor> inputs,
                            std::vector<synTensor> outputs,
                            synConcatenateParams params,
                            std::string node_name) {
    std::string guid = "concat";
    AddNode_IOP<synConcatenateParams>(inputs, outputs, params, guid, node_name);
  }

  inline void AddNodeSplit(std::vector<synTensor> inputs,
                           std::vector<synTensor> outputs,
                           synSplitParams params,
                           std::string node_name) {
    std::string guid = "split";
    AddNode_IOP<synSplitParams>(inputs, outputs, params, guid, node_name);
  }

  inline void AddNodeSlice(std::vector<synTensor> inputs,
                           std::vector<synTensor> outputs,
                           synSliceParamsV2 params,
                           std::string node_name) {
    std::string guid = "slice";
    AddNode_IOP<synSliceParamsV2>(inputs, outputs, params, guid, node_name);
  }

  inline void AddNodeSqueeze(std::vector<synTensor> inputs,
                             std::vector<synTensor> outputs,
                             synSqueezeParams params,
                             std::string node_name) {
    std::string guid = "squeeze";
    AddNode_IOP<synSqueezeParams>(inputs, outputs, params, guid, node_name);
  }

  inline void AddNodeTopK(std::vector<synTensor> inputs,
                          std::vector<synTensor> outputs,
                          ns_TopkNodeV2::ParamsV4 params,
                          std::string node_name) {
    std::string guid = "topk";
    AddNode_IOP<ns_TopkNodeV2::ParamsV4>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeMultinomial(std::vector<synTensor> inputs,
                                 std::vector<synTensor> outputs,
                                 ns_RandomMultinomial::ParamsV2 params,
                                 std::string node_name) {
    std::string guid = "random_multinomial_pt_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_RandomMultinomial::ParamsV2>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeSoftmax(std::vector<synTensor> inputs,
                             std::vector<synTensor> outputs,
                             ns_Softmax::Params params,
                             std::string node_name) {
    std::string guid = "softmax_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_Softmax::Params>(inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeRmsNorm(std::vector<synTensor> inputs,
                             std::vector<synTensor> outputs,
                             ns_LayerNormKernel::Params params,
                             std::string node_name) {
    std::string guid = "rms_norm_ex_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_LayerNormKernel::Params>(
        inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeRope(std::vector<synTensor> inputs,
                          std::vector<synTensor> outputs,
                          ns_RoPESt2::ParamsV2 params,
                          std::string node_name) {
    std::string guid = "rotary_pos_embedding_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_RoPESt2::ParamsV2>(inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  inline void AddNodeSdpaRecomp(std::vector<synTensor> inputs,
                                std::vector<synTensor> outputs,
                                ns_Sdpa::ParamsV3 params,
                                std::string node_name) {
    std::string guid = "sdpa_recomp_fwd_" + guid_dtype<T>();
    AddNode_IOP<ns_Sdpa::ParamsV3>(inputs, outputs, params, guid, node_name);
  }

  template <typename T>
  // ns_QuantizationPerTensor ignored
  inline void AddNodeQuantizePerTensor(std::vector<synTensor> inputs,
                                       std::vector<synTensor> outputs,
                                       std::string node_name) {
    std::string guid = "quantize_per_tensor_" + guid_dtype<T>();
    AddNode_IO(inputs, outputs, guid, node_name);
  }

  template <typename T>
  inline void AddNodeMoeForward(std::vector<synTensor> inputs,
                                std::vector<synTensor> outputs,
                                std::shared_ptr<ns_MoeKernel::ParamsV3> params,
                                std::string node_name) {
    std::string guid = "moe_" + guid_dtype<T>();
    AddNode_IOP<ns_MoeKernel::ParamsV3>(
        inputs, outputs, *params, guid, node_name);
  }

  synTensor cloneTensor(std::string name, synTensor base, synDataType type) {
    synTensorGeometry geometry;
    synTensorGetGeometry(base, &geometry, synGeometrySizes);

    std::vector<int64_t> dims;
    for (unsigned int i = 0; i < geometry.dims; i++) {
      dims.push_back(geometry.sizes[geometry.dims - 1 - i]);
    }

    return createTensorNoPresist(name, type, dims);
  }

  template <typename T>
  void AddNodeFusedFP8Gemm(std::vector<synTensor> inputs,
                           std::vector<synTensor> outputs,
                           synGEMMParams params,
                           std::string node_name) {
    synTensorDeviceFullLayout x_layout;
    synTensorDeviceFullLayout y_layout;
    synTensorGetDeviceFullLayout(inputs[0], &x_layout);
    synTensorGetDeviceFullLayout(inputs[1], &y_layout);

    bool cast_x = (x_layout.deviceDataType != syn_type_fp8_143);
    bool cast_y = (y_layout.deviceDataType != syn_type_fp8_143);
    ns_CastKernel::Params cast_to_fp8_params;
    synTensor x_tensor = inputs[0];
    synTensor y_tensor = inputs[1];

    cast_to_fp8_params.round_mode = CAST_ROUND_HALF_NE;
    if (cast_x) {
      x_tensor = cloneTensor(node_name + "_x", inputs[0], syn_type_fp8_143);
      std::vector<synTensor> cast_ins = {inputs[0], inputs[2]};
      std::vector<synTensor> cast_outs = {x_tensor};
      AddNodeConvertToFP8<T>(
          cast_ins, cast_outs, cast_to_fp8_params, node_name + "_cast_x");
    }
    if (cast_y) {
      y_tensor = cloneTensor(node_name + "_y", inputs[1], syn_type_fp8_143);
      std::vector<synTensor> cast_ins = {inputs[1], inputs[3]};
      std::vector<synTensor> cast_outs = {y_tensor};
      AddNodeConvertToFP8<T>(
          cast_ins, cast_outs, cast_to_fp8_params, node_name + "_cast_y");
    }

    std::vector<synTensor> gemm_ins;
    gemm_ins.push_back(x_tensor);
    gemm_ins.push_back(y_tensor);
    if (!cast_x) {
      gemm_ins.push_back(inputs[2]);
    }
    if (!cast_y) {
      gemm_ins.push_back(inputs[3]);
    }
    AddNodeFP8Gemm<T>(gemm_ins, outputs, params, node_name);
  }
};

}  // namespace custom_kernel
