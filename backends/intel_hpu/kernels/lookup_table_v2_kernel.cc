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

#include "funcs.h"
#include "hpu_operator.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "synapse_common_types.h"

namespace custom_kernel {

class Embedding : public HpuOperator {
 public:
  Embedding(synDataType dtype) : HpuOperator("gather_fwd_"), dtype_(dtype) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               int axis_dim) {
    std::vector<synTensor> inputs;
    inputs.push_back(createTensor(ins[0].size(), dtype_, ins[0], true, "weight"));
    inputs.push_back(
        createTensor(ins[1].size(), syn_type_int32, ins[1], true, "inputx"));

    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), dtype_, outs[0], true, "output")};

    ns_GatherKernel::Params params;
    params.axis = ins[0].size() - 1 - axis_dim;

    if (dtype_ == syn_type_fp16) {
      guid_ = guid_ + "f16";
    } else if (dtype_ == syn_type_bf16) {
      guid_ = guid_ + "bf16";
    } else if (dtype_ == syn_type_single) {
      guid_ = guid_ + "f32";
    }

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs,
                                     ins.size(),
                                     1,
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "gather",
                                     nullptr,
                                     nullptr);
    CHKSTATUS("synNodeCreate reshape failed!");
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void EmbeddingKernel(const Context& dev_ctx,
                     const phi::DenseTensor& inputx,
                     const phi::DenseTensor& weight,
                     int64_t padding_idx,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  synDataType dtype = syn_type_na;
  if (std::is_same<T, phi::dtype::float16>::value) {
    dtype = syn_type_fp16;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    dtype = syn_type_bf16;
  } else if (std::is_same<T, phi::dtype::float8_e4m3fn>::value) {
    dtype = syn_type_fp8_143;
  } else if (std::is_same<T, float>::value) {
    dtype = syn_type_single;
  }

  Embedding op(dtype);
  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(weight.dims());
  std::vector<int64_t> index_dims = phi::vectorize<int64_t>(inputx.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

  op.AddNode({x_dims, index_dims}, {outputs_dim}, 0);
  op.Compile();

  // std::map<std::string, uint64_t> tensors;
  // tensors["weight"] = reinterpret_cast<uint64_t>(weight.data<T>());
  // // TODO, index.dtype == int64
  // tensors["inputx"] = reinterpret_cast<uint64_t>(inputx.data<int32_t>());
  // tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
  // op.Execute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(embedding,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::EmbeddingKernel,
                          float,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
