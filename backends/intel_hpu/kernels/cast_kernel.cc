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
#include "utils/utills.h"
namespace custom_kernel {

struct CastParams {
  phi::DataType src_type;
  phi::DataType dst_type;
};

class Cast : public HpuOperator {
 public:
  Cast(synDataType src_type, synDataType dst_type)
      : HpuOperator("cast"), src_type_(src_type), dst_type_(dst_type) {}
  void AddNode(const std::vector<DIMS>& ins, const std::vector<DIMS>& outs) {
    std::vector<synTensor> inputs;
    inputs.push_back(createTensor(ins[0].size(), src_type_, ins[0], true, "x"));

    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), dst_type_, outs[0], true, "output")};

    if (src_type_ == syn_type_fp16) {
      guid_ = guid_ + "_f16_to";
    } else if (src_type_ == syn_type_bf16) {
      guid_ = guid_ + "_bf16_to";
    } else if (src_type_ == syn_type_single) {
      guid_ = guid_ + "_f32_to";
    } else if (src_type_ == syn_type_int32) {
      guid_ = guid_ + "_i32_to";
    }

    if (dst_type_ == syn_type_fp16) {
      guid_ = guid_ + "_f16";
    } else if (dst_type_ == syn_type_bf16) {
      guid_ = guid_ + "_bf16";
    } else if (dst_type_ == syn_type_single) {
      guid_ = guid_ + "_f32";
    }else if (dst_type_ == syn_type_int32) {
      guid_ = guid_ + "_i32";
    }

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs,
                                     ins.size(),
                                     1,
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     "CAST",
                                     nullptr,
                                     nullptr);
    CHKSTATUS("synNodeCreate reshape failed!");
  }

 protected:
  synDataType src_type_;
  synDataType dst_type_;
};

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  if (x.dtype() == dtype) {
    dev_ctx.template Alloc<T>(out);
    TensorCopy(dev_ctx, x, true, out);
    return;
  }

  std::map<std::string, uint64_t> tensors;

  if (dtype == phi::DataType::FLOAT32) {
    dev_ctx.template Alloc<float>(out);
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<float>());
  } else if (dtype == phi::DataType::FLOAT64) {
    dev_ctx.template Alloc<double>(out);
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<double>());
  } else if (dtype == phi::DataType::FLOAT16) {
    dev_ctx.template Alloc<phi::dtype::float16>(out);
    tensors["output"] =
        reinterpret_cast<uint64_t>(out->data<phi::dtype::float16>());
  } else if (dtype == phi::DataType::BFLOAT16) {
    dev_ctx.template Alloc<phi::dtype::bfloat16>(out);
    tensors["output"] =
        reinterpret_cast<uint64_t>(out->data<phi::dtype::bfloat16>());
  } else if (dtype == phi::DataType::INT16) {
    dev_ctx.template Alloc<int16_t>(out);
  } else if (dtype == phi::DataType::INT32) {
    dev_ctx.template Alloc<int32_t>(out);
  } else if (dtype == phi::DataType::INT64) {
    dev_ctx.template Alloc<int64_t>(out);
  } else if (dtype == phi::DataType::BOOL) {
    dev_ctx.template Alloc<bool>(out);
  } else if (dtype == phi::DataType::UINT8) {
    dev_ctx.template Alloc<uint8_t>(out);
  } else if (dtype == phi::DataType::INT8) {
    dev_ctx.template Alloc<int8_t>(out);
  } else {
    phi::errors::InvalidArgument("Unsupported cast dtype %s", dtype);
  }

  OpCacheOperator op_info;

  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());
  CastParams params;
  params.src_type = x.dtype();
  params.dst_type = dtype;

  op_info.prepareOpInfo<T, CastParams>("CastKernel", {x_dims}, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    auto dst_syn_type = PDDataTypeToSynDataType(dtype);
    auto src_syn_type = PDDataTypeToSynDataType(x.dtype());
    Cast op(src_syn_type, dst_syn_type);

    op.AddNode({x_dims}, {outputs_dim});
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cast,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::CastKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
