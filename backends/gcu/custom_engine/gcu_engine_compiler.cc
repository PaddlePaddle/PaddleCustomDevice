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

#include "custom_engine/gcu_engine_compiler.h"

#include "custom_engine/ir_translator/translator_registry.h"
#include "paddle/common/flags.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

COMMON_DECLARE_bool(print_ir);

namespace custom_engine {
class GCUEngineCompiler::GCUEngineCompilerImpl {
 public:
  GCUEngineCompilerImpl(
      const phi::KernelContext& kernel_context,
      pir::Operation* op,
      const std::vector<pir::Value>& engine_inputs,
      const std::vector<pir::Value>& engine_outputs,
      const std::unordered_map<pir::Value, std::vector<phi::DenseTensor*>>&
          engine_value_to_tensors,
      const std::unordered_map<pir::Value, std::vector<std::string>>&
          engine_value_to_var_names,
      const std::string& engine_key)
      : kernel_context_(kernel_context),
        op_(op),
        engine_inputs_(engine_inputs),
        engine_outputs_(engine_outputs),
        engine_value_to_tensors_(engine_value_to_tensors),
        engine_value_to_var_names_(engine_value_to_var_names),
        engine_key_(engine_key) {
    Init();
  }

  ~GCUEngineCompilerImpl() {}

  void Init();

  void Compile(GCUEngine* gcu_engine);

 private:
  void CreateInputs();
  void CreateConstOrParameter();
  void MapInnerOutputValues(const pir::Operation* yield_op);
  void SetGraphOutputs();
  void ConvertGraph();

  phi::KernelContext kernel_context_;
  pir::Operation* op_;  // Not owned

  std::string engine_key_;

  std::vector<pir::Value> engine_inputs_;
  std::vector<pir::Value> engine_outputs_;
  std::vector<pir::Value> engine_inner_outputs_;
  std::unordered_map<pir::Value, std::vector<phi::DenseTensor*>>
      engine_value_to_tensors_;
  std::unordered_map<pir::Value, std::vector<std::string>>
      engine_value_to_var_names_;

  pir::Block* block_;
  std::vector<const phi::DenseTensor*> inputs_;
  std::vector<phi::DenseTensor*> outputs_;
  std::vector<const phi::DenseTensor*> device_constants_;
  std::vector<phi::DenseTensor*> parameters_;
  std::unordered_set<pir::Value> uninitialized_const_or_params_;

  // for GCU graph
  GcuBuilderPtr builder_ = nullptr;
  std::unordered_map<phi::DenseTensor*, GcuOpPtr> gcu_op_cache_;
};

GCUEngineCompiler::GCUEngineCompiler::GCUEngineCompiler(
    const phi::KernelContext& kernel_context,
    pir::Operation* op,
    const std::vector<pir::Value>& engine_inputs,
    const std::vector<pir::Value>& engine_outputs,
    const std::unordered_map<pir::Value, std::vector<phi::DenseTensor*>>&
        engine_value_to_tensors,
    const std::unordered_map<pir::Value, std::vector<std::string>>&
        engine_value_to_var_names,
    const std::string& engine_key) {
  impl_ = std::make_shared<GCUEngineCompilerImpl>(kernel_context,
                                                  op,
                                                  engine_inputs,
                                                  engine_outputs,
                                                  engine_value_to_tensors,
                                                  engine_value_to_var_names,
                                                  engine_key);
}

void GCUEngineCompiler::Compile(GCUEngine* gcu_engine) {
  impl_->Compile(gcu_engine);
}

void GCUEngineCompiler::GCUEngineCompilerImpl::Init() {
  pir::Region& region = op_->region(0);
  PADDLE_ENFORCE_EQ(
      region.empty(),
      false,
      ::common::errors::Unavailable(
          "Required CustomEngineOp's region must not be emptpy."));
  block_ = &(region.front());

  //   inputs_ = kernel_context_.InputsBetween<phi::DenseTensor>(
  //       size_t(0), kernel_context_.InputsSize());

  //   auto outputs = kernel_context_.MutableOutputBetween<phi::DenseTensor>(
  //       size_t(0), kernel_context_.OutputsSize());
  //   outputs_.assign(outputs.begin(), outputs.end());

  for (size_t i = 0; i < engine_inputs_.size(); ++i) {
    PADDLE_ENFORCE_GT(engine_value_to_tensors_.count(engine_inputs_[i]),
                      0,
                      common::errors::PreconditionNotMet(
                          "Input[%zu] is not in value map", i));
    inputs_.emplace_back(engine_value_to_tensors_.at(engine_inputs_[i]).at(0));
  }

  for (size_t i = 0; i < engine_outputs_.size(); ++i) {
    PADDLE_ENFORCE_GT(engine_value_to_tensors_.count(engine_outputs_[i]),
                      0,
                      common::errors::PreconditionNotMet(
                          "Output[%zu] is not in value map", i));
    outputs_.emplace_back(
        engine_value_to_tensors_.at(engine_outputs_[i]).at(0));
  }

  builder_ = std::make_shared<GcuBuilder>();
  PADDLE_ENFORCE_NOT_NULL(
      builder_, "Faild to reate gcu builder for %s.", engine_key_.c_str());
  builder_->SetShapeInference(true);

  VLOG(3) << "GCUEngineCompiler Init successfully.";
}

void GCUEngineCompiler::GCUEngineCompilerImpl::Compile(GCUEngine* gcu_engine) {
  PADDLE_ENFORCE_NOT_NULL(gcu_engine,
                          "The return GCUEngine memory is not allocated.");
  VLOG(3) << "Compile for " << engine_key_;
  ConvertGraph();

  auto hlir_module = builder_->GetModule();
  VLOG(3) << "Compiler begin to CompileHLIR for " << engine_key_;
  topsExecutable_t tops_executable =
      custom_engine::CompileTopsExecutable(hlir_module);

  VLOG(3) << "Compiler CompileHLIR end for " << engine_key_;
  std::vector<const phi::DenseTensor*> total_inputs(inputs_);
  total_inputs.insert(
      total_inputs.end(), parameters_.begin(), parameters_.end());
  total_inputs.insert(
      total_inputs.end(), device_constants_.begin(), device_constants_.end());

  std::vector<phi::DenseTensor*> total_outputs(outputs_);
  //   // not necessary
  //   total_outputs.insert(total_outputs.end(), parameters_.begin(),
  //                        parameters_.end());

  gcu_engine->Init(engine_key_, tops_executable, total_inputs, total_outputs);
  VLOG(3) << "Generate GCUEngine for " << engine_key_;
  return;
}

void GCUEngineCompiler::GCUEngineCompilerImpl::CreateInputs() {
  for (size_t i = 0; i < engine_inputs_.size(); ++i) {
    auto tensor = engine_value_to_tensors_.at(engine_inputs_[i]).at(0);

    auto ptype = custom_engine::ConvertFromPhiDataType(tensor->dtype());
    std::vector<int64_t> dims = common::vectorize(tensor->dims());
    builder::Type input_type(dims, ptype);
    gcu_op_cache_[tensor] =
        std::make_shared<GcuOp>(builder_->CreateInput(input_type));
    VLOG(6) << "Create gcu builder input[" << i
            << "]: " << engine_value_to_var_names_.at(engine_inputs_[i]).at(0)
            << ", dims:" << custom_engine::VectorToStr<int64_t>(dims);
  }
}

void GCUEngineCompiler::GCUEngineCompilerImpl::CreateConstOrParameter() {
  auto get_const_and_parameters_func =
      [](const std::list<pir::Operation*>& graph_ops,
         std::vector<const pir::Operation*>& constants,
         std::vector<const pir::Operation*>& parameters) -> void {
    for (const auto* op : graph_ops) {
      std::string op_name = op->name();
      auto op_attributes = op->attributes();
      if (op->HasAttribute("op_name")) {
        op_name = op->attribute<pir::StrAttribute>("op_name").AsString();
      }
      if (op->isa<pir::ParameterOp>()) {
        parameters.emplace_back(op);
      } else if (op_name == pir::ConstantOp::name()) {
        constants.emplace_back(op);
      }
    }
  };

  std::list<pir::Operation*> current_graph_ops = block_->ops();
  pir::Block* parent_block = op_->GetParentProgram()->block();
  std::list<pir::Operation*> parent_graph_ops = parent_block->ops();

  std::vector<const pir::Operation*> current_graph_constant_ops;
  std::vector<const pir::Operation*> current_graph_parameters;
  std::vector<const pir::Operation*> parent_graph_constant_ops;
  std::vector<const pir::Operation*> parent_graph_parameters;

  get_const_and_parameters_func(
      current_graph_ops, current_graph_constant_ops, current_graph_parameters);
  get_const_and_parameters_func(
      parent_graph_ops, parent_graph_constant_ops, parent_graph_parameters);

  VLOG(6) << "CreateConstOrParameter get_const_and_parameters_func, current "
             "graph parameters size:"
          << current_graph_parameters.size()
          << ", constant op size:" << current_graph_constant_ops.size();

  VLOG(6) << "CreateConstOrParameter get_const_and_parameters_func, parent "
             "graph parameters size:"
          << parent_graph_parameters.size()
          << ", constant op size:" << parent_graph_constant_ops.size();

  std::vector<const pir::Operation*> total_parameters(current_graph_parameters);
  total_parameters.insert(total_parameters.end(),
                          parent_graph_parameters.begin(),
                          parent_graph_parameters.end());

  std::vector<const pir::Operation*> total_constants(
      current_graph_constant_ops);
  total_constants.insert(total_constants.end(),
                         parent_graph_constant_ops.begin(),
                         parent_graph_constant_ops.end());

  for (const auto* op : total_parameters) {
    auto value = op->result(0);
    auto tensor = engine_value_to_tensors_.at(value).at(0);
    if (gcu_op_cache_.count(tensor) > 0) {
      continue;
    }
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<pir::StrAttribute>()
                          .AsString();
    if (!(tensor->initialized())) {
      uninitialized_const_or_params_.insert(value);
      VLOG(6) << "Skip uninitialized parameter, name:" << param_name
              << ", dims: [" << tensor->dims() << "].";
      continue;
    }

    auto ptype = custom_engine::ConvertFromPhiDataType(tensor->dtype());
    std::vector<int64_t> dims = common::vectorize(tensor->dims());
    builder::Type gcu_op_type(dims, ptype);
    gcu_op_cache_[tensor] =
        std::make_shared<GcuOp>(builder_->CreateInput(gcu_op_type));
    parameters_.emplace_back(tensor);
    VLOG(6) << "Create gcu builder Parameter:" << param_name
            << ", dims:" << custom_engine::VectorToStr<int64_t>(dims);
  }

  for (const auto* op : total_constants) {
    if (op->isa<pir::ConstantTensorOp>()) {
      auto const_name = op->dyn_cast<pir::ConstantTensorOp>().tensor_name();
      auto value = op->result(0);
      auto tensor = engine_value_to_tensors_.at(value).at(0);
      if (gcu_op_cache_.count(tensor) > 0) {
        continue;
      }
      if (!(tensor->initialized())) {
        uninitialized_const_or_params_.insert(value);
        VLOG(6) << "Skip uninitialized constant, name:" << const_name
                << ", dims: [" << tensor->dims() << "].";
        continue;
      }
      auto ptype = custom_engine::ConvertFromPhiDataType(tensor->dtype());
      std::vector<int64_t> dims = common::vectorize(tensor->dims());
      builder::Type gcu_op_type(dims, ptype);
      builder::Op const_op;
      if (tensor->place().GetType() == phi::AllocationType::CPU) {
        const_op = builder::Const(
            builder_, static_cast<void*>(tensor->data()), gcu_op_type);
      } else {
        const_op = builder_->CreateInput(gcu_op_type);
        device_constants_.emplace_back(tensor);
      }

      gcu_op_cache_[tensor] = std::make_shared<GcuOp>(const_op);
      VLOG(6) << "Create gcu builder Constant:" << const_name
              << ", dims:" << custom_engine::VectorToStr<int64_t>(dims);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Only support ConstantTensorOp, name:%s, id:%ld.",
          op->name().c_str(),
          op->name()));
    }
  }
}

void GCUEngineCompiler::GCUEngineCompilerImpl::MapInnerOutputValues(
    const pir::Operation* yield_op) {
  std::vector<std::vector<GcuOpPtr>> input_gcu_ops;
  size_t input_num = yield_op->num_operands();
  VLOG(6) << "MapOutputValues for yeild op:" << yield_op->name()
          << ", input num:" << input_num;
  PADDLE_ENFORCE_EQ(input_num,
                    engine_outputs_.size(),
                    common::errors::PreconditionNotMet(
                        "Output num check failed, except:%zu, but get:%zu",
                        engine_outputs_.size(),
                        input_num));
  for (size_t i = 0; i < input_num; ++i) {
    auto value = yield_op->operand_source(i);
    PADDLE_ENFORCE_GT(
        engine_value_to_tensors_.count(value),
        0,
        common::errors::PreconditionNotMet(
            "Input[%zu] value of yeild is not in engine_value_to_tensors_", i));
    PADDLE_ENFORCE_GT(
        engine_value_to_var_names_.count(value),
        0,
        common::errors::PreconditionNotMet(
            "Input[%zu] value of yeild is not in engine_value_to_var_names_",
            i));

    engine_inner_outputs_.emplace_back(value);
  }
}

void GCUEngineCompiler::GCUEngineCompilerImpl::SetGraphOutputs() {
  std::vector<GcuOp> graph_outputs;
  for (size_t i = 0; i < engine_inner_outputs_.size(); ++i) {
    auto tensors = engine_value_to_tensors_.at(engine_inner_outputs_[i]);
    PADDLE_ENFORCE_EQ(tensors.size(),
                      1,
                      common::errors::PreconditionNotMet(
                          "Only support one tensor now, but get %zu, "
                          "output_index:%zu",
                          tensors.size(),
                          i));

    auto tensor = tensors.at(0);
    auto inner_value_name =
        engine_value_to_var_names_.at(engine_inner_outputs_[i]).at(0);
    auto external_value_name =
        engine_value_to_var_names_.at(engine_outputs_[i]).at(0);
    PADDLE_ENFORCE_GT(
        gcu_op_cache_.count(tensor),
        0,
        common::errors::PreconditionNotMet(
            "Output[%zu] is not generated in gcu_op map, value name:%s",
            i,
            inner_value_name.c_str()));
    graph_outputs.emplace_back(*(gcu_op_cache_.at(tensor)));

    // set output shapes
    auto gcu_shape = gcu_op_cache_.at(tensor)->GetType().GetShape();
    tensor->Resize(common::make_ddim(gcu_shape));
    outputs_[i]->Resize(common::make_ddim(gcu_shape));
    // *(outputs_[i]) = *tensor;
    VLOG(6) << "Found gcu builder output[" << i << "]: " << inner_value_name
            << ", external var name:" << external_value_name
            << ", dims:" << tensor->dims();
  }
  builder_->SetOutput(graph_outputs);
}

void GCUEngineCompiler::GCUEngineCompilerImpl::ConvertGraph() {
  VLOG(3) << "ConvertGraph for " << engine_key_;
  if (FLAGS_print_ir) {
    std::cout << "IR Before conversion = " << *block_ << std::endl;
  }

  VLOG(3) << "Create inputs node for " << engine_key_;
  CreateInputs();
  CreateConstOrParameter();
  //   builder_->Dump();

  VLOG(3) << "Convert calc ops for " << engine_key_;
  // NOTES: Consider the subgraph to be topologically sorted.
  std::list<pir::Operation*> graph_ops = block_->ops();
  for (const auto* op : graph_ops) {
    if (op->isa<pir::YieldOp>()) {
      MapInnerOutputValues(op);
      continue;
    }
    std::string op_name = op->name();
    auto op_attributes = op->attributes();
    if (op->HasAttribute("op_name")) {
      op_name = op->attribute<pir::StrAttribute>("op_name").AsString();
    }

    OpTranslateFunc convert_func =
        TranslatorRegistry::Instance().Get(OpTranslateFuncKey(op_name));
    PADDLE_ENFORCE_NOT_NULL(convert_func);

    // inputs
    std::vector<std::vector<GcuOpPtr>> input_gcu_ops;
    size_t input_num = op->num_operands();
    VLOG(6) << "Get input_gcu_ops for " << op_name << ", num:" << input_num;
    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand_source(i);
      bool invalid =
          (!(value) || (uninitialized_const_or_params_.count(value) > 0));
      if (invalid) {
        std::vector<GcuOpPtr> invalid_ops = {std::make_shared<GcuOp>()};
        input_gcu_ops.emplace_back(invalid_ops);
        VLOG(3) << "Input " << i
                << " is a NULL VALUE or uninitialized, maybe it is an "
                   "optional input or zero tensor, op_name:"
                << op_name;
        continue;
      }

      PADDLE_ENFORCE_GT(
          engine_value_to_tensors_.count(value),
          0,
          common::errors::PreconditionNotMet(
              "Input[%zu] value is not in engine_value_to_tensors_", i));
      PADDLE_ENFORCE_GT(
          engine_value_to_var_names_.count(value),
          0,
          common::errors::PreconditionNotMet(
              "Input[%zu] value is not in engine_value_to_var_names_", i));

      std::vector<GcuOpPtr> gcu_ops;
      auto tensors = engine_value_to_tensors_.at(value);
      auto var_names = engine_value_to_var_names_.at(value);

      for (size_t n = 0; n < tensors.size(); ++n) {
        PADDLE_ENFORCE_GT(
            gcu_op_cache_.count(tensors[n]),
            0,
            common::errors::PreconditionNotMet(
                "Input[%zu][%zu] is not generated in gcu_op map, name: %s",
                i,
                n,
                var_names.at(n)));
        auto gcu_op_in = gcu_op_cache_.at(tensors[n]);
        gcu_ops.emplace_back(gcu_op_in);
        VLOG(6) << "op_name:" << op_name << ", inputs[" << i << "][" << n
                << "], var name:" << var_names.at(n) << ", dims:"
                << custom_engine::VectorToStr<int64_t>(
                       gcu_op_in->GetType().GetShape());
      }
      input_gcu_ops.emplace_back(gcu_ops);
    }

    // convert
    VLOG(6) << "Start to convert for " << op_name;
    GcuOpPtr gcu_op = convert_func(builder_, op, input_gcu_ops);
    VLOG(6) << "End of conversion for " << op_name;
    if (gcu_op == nullptr) {
      VLOG(6) << "Skip conversion for " << op_name;
      continue;
    }

    bool is_tuple_out = gcu_op->GetType().IsTuple();
    if (is_tuple_out) {
      size_t gcu_output_num = gcu_op->GetType().GetTupleSize();
      size_t output_num = op->num_results();
      PADDLE_ENFORCE_EQ(
          gcu_output_num,
          output_num,
          common::errors::PreconditionNotMet("Output num check failed, op: %s",
                                             op_name.c_str()));

      for (size_t i = 0; i < output_num; ++i) {
        auto out_value = op->result(i);
        if (engine_value_to_tensors_.count(out_value) == 0) {
          VLOG(3) << "Output GetTupleElement for " << op_name
                  << ", output index:" << i
                  << " not found, maybe a reserve space.";
          continue;
        }
        auto tensors = engine_value_to_tensors_.at(out_value);
        PADDLE_ENFORCE_EQ(tensors.size(),
                          1,
                          common::errors::PreconditionNotMet(
                              "Only support one tensor now, but get %zu, op: "
                              "%s, output_index:%zu",
                              tensors.size(),
                              op_name.c_str(),
                              i));

        auto tensor = tensors.at(0);
        auto gcu_op_out =
            std::make_shared<GcuOp>(builder::GetTupleElement(*gcu_op, i));
        gcu_op_cache_[tensor] = gcu_op_out;
        VLOG(6) << "Output GetTupleElement for " << op_name
                << ", output index:" << i
                << ", name:" << engine_value_to_var_names_.at(out_value).at(0)
                << ", dims:"
                << custom_engine::VectorToStr<int64_t>(
                       gcu_op_out->GetType().GetShape());
      }
    } else {
      if (op->num_results() == 1) {
        auto out_value = op->result(0);
        auto tensors = engine_value_to_tensors_.at(out_value);
        PADDLE_ENFORCE_EQ(tensors.size(),
                          1,
                          common::errors::PreconditionNotMet(
                              "Output num should be one, but get %zu, op: "
                              "%s, output_index:%zu",
                              tensors.size(),
                              op_name.c_str()));
        gcu_op_cache_[tensors.at(0)] = gcu_op;
        VLOG(6) << "Output set for " << op_name
                << ", name:" << engine_value_to_var_names_.at(out_value).at(0)
                << ", dims:"
                << custom_engine::VectorToStr<int64_t>(
                       gcu_op->GetType().GetShape());
      } else {
        VLOG(6) << "Op " << op_name << " does not have any output value.";
      }
    }
  }  // end of for (const auto* op : graph_ops)
  // outputs
  SetGraphOutputs();
  if (FLAGS_print_ir) {
    std::cout << "IR After conversion = " << std::endl;
    builder_->Dump();
  }
}

}  // namespace custom_engine
