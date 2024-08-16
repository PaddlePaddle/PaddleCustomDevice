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

struct ReduceParams {
  ns_Reduction::Params params;
  bool keep_dim;
  phi::IntArray dims;
};

class Reduce : public HpuOperator {
 public:
  Reduce(synDataType dtype, std::string op)
      : HpuOperator("reduce"), op_(op), dtype_(dtype) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               ReduceParams params) {
    std::vector<synTensor> inputs;
    inputs.push_back(createTensor(ins[0].size(), dtype_, ins[0], true, "x"));

    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), dtype_, outs[0], true, "output")};

    std::string name = guid_ + op_;
    guid_ = guid_ + "_" + op_ + "_fwd_";
    if (dtype_ == syn_type_fp16) {
      guid_ = guid_ + "f16";
    } else if (dtype_ == syn_type_bf16) {
      guid_ = guid_ + "bf16";
    } else if (dtype_ == syn_type_single) {
      guid_ = guid_ + "f32";
    }
    LOG(INFO) << "reduce dim = " << params.params.reductionDimension;
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs,
                                     ins.size(),
                                     1,
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     name.c_str(),
                                     nullptr,
                                     nullptr);
    CHKSTATUS("synNodeCreate reshape failed!");
  }

 protected:
  synDataType dtype_;
  std::string op_;
};

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (dims.size() == 1) {
    OpCacheOperator op_info;
    std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
    std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

    ReduceParams params;
    params.keep_dim = keep_dim;
    auto rank = static_cast<int32_t>(x.dims().size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(x.dims().size()));
    params.params.reductionDimension = rank - 1 - dim;
    op_info.prepareOpInfo<T, ReduceParams>("MeanKernel", {x_dims}, &params);
    auto recipe = op_info.GetRecipe();
    if (recipe == nullptr) {
      Reduce op(op_info.datatype_, "mean");
      op.AddNode({x_dims}, {outputs_dim}, params);
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    }

    std::map<std::string, uint64_t> tensors;
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
    RecipeRunner runner(recipe);
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  }
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (dims.size() == 1) {
    OpCacheOperator op_info;
    std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
    std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

    ReduceParams params;
    params.keep_dim = keep_dim;
    auto rank = static_cast<int32_t>(x.dims().size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(x.dims().size()));
    params.params.reductionDimension = rank - 1 - dim;
    op_info.prepareOpInfo<T, ReduceParams>("MaxKernel", {x_dims}, &params);
    auto recipe = op_info.GetRecipe();
    if (recipe == nullptr) {
      Reduce op(op_info.datatype_, "max");
      op.AddNode({x_dims}, {outputs_dim}, params);
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    }

    std::map<std::string, uint64_t> tensors;
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
    RecipeRunner runner(recipe);
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  }
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (dims.size() == 1) {
    OpCacheOperator op_info;
    std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
    std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

    ReduceParams params;
    params.keep_dim = keep_dim;
    auto rank = static_cast<int32_t>(x.dims().size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(x.dims().size()));
    params.params.reductionDimension = rank - 1 - dim;
    op_info.prepareOpInfo<T, ReduceParams>("MinKernel", {x_dims}, &params);
    auto recipe = op_info.GetRecipe();
    if (recipe == nullptr) {
      Reduce op(op_info.datatype_, "min");
      op.AddNode({x_dims}, {outputs_dim}, params);
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    }

    std::map<std::string, uint64_t> tensors;
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
    RecipeRunner runner(recipe);
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  }
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (dims.size() == 1) {
    OpCacheOperator op_info;
    std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
    std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

    ReduceParams params;
    params.keep_dim = keep_dim;
    auto rank = static_cast<int32_t>(x.dims().size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(x.dims().size()));
    params.params.reductionDimension = rank - 1 - dim;

    op_info.prepareOpInfo<T, ReduceParams>("SumKernel", {x_dims}, &params);
    auto recipe = op_info.GetRecipe();
    if (recipe == nullptr) {
      Reduce op(op_info.datatype_, "sum");
      op.AddNode({x_dims}, {outputs_dim}, params);
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    }

    std::map<std::string, uint64_t> tensors;
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
    RecipeRunner runner(recipe);
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  }
}

template <typename T, typename Context>
void ProdKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                phi::DataType out_dtype,
                bool keep_dim,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (dims.size() == 1) {
    OpCacheOperator op_info;
    std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
    std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

    ReduceParams params;
    params.keep_dim = keep_dim;
    auto rank = static_cast<int32_t>(x.dims().size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(x.dims().size()));
    params.params.reductionDimension = rank - 1 - dim;
    op_info.prepareOpInfo<T, ReduceParams>("ProdKernel", {x_dims}, &params);
    auto recipe = op_info.GetRecipe();
    if (recipe == nullptr) {
      Reduce op(op_info.datatype_, "prod");
      op.AddNode({x_dims}, {outputs_dim}, params);
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    }

    std::map<std::string, uint64_t> tensors;
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
    RecipeRunner runner(recipe);
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MeanKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(max,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(prod,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ProdKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(min,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MinKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}