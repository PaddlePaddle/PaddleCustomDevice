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

#include <glog/logging.h>

#include <bitset>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// inline auto kCanRunGcuAttr = paddle::dialect::kCanRunGcuAttr;
inline const char kCanRunGcuAttr[] = "__l_gcu__";

#define DEFINE_GENERAL_PATTERN(OpName, OpType)                            \
  class OpName##OpPattern : public pir::OpRewritePattern<OpType> {        \
   public:                                                                \
    using pir::OpRewritePattern<OpType>::OpRewritePattern;                \
    bool MatchAndRewrite(OpType op,                                       \
                         pir::PatternRewriter &rewriter) const override { \
      if (op->HasAttribute(kCanRunGcuAttr) &&                             \
          op->attribute<pir::BoolAttribute>(kCanRunGcuAttr).data()) {     \
        return false;                                                     \
      }                                                                   \
      op->set_attribute(kCanRunGcuAttr, rewriter.bool_attr(true));        \
      return true;                                                        \
    }                                                                     \
  };

// Special builtin ops
DEFINE_GENERAL_PATTERN(Combine, pir::CombineOp)
DEFINE_GENERAL_PATTERN(Split, pir::SplitOp)
DEFINE_GENERAL_PATTERN(Parameter, pir::ParameterOp)
DEFINE_GENERAL_PATTERN(Constant, pir::ConstantOp)
DEFINE_GENERAL_PATTERN(ConstantTensor, pir::ConstantTensorOp)

// Normal ops
DEFINE_GENERAL_PATTERN(Matmul, paddle::dialect::MatmulOp)
DEFINE_GENERAL_PATTERN(Add, paddle::dialect::AddOp)
DEFINE_GENERAL_PATTERN(Add_, paddle::dialect::Add_Op)
DEFINE_GENERAL_PATTERN(Subtract, paddle::dialect::SubtractOp)
DEFINE_GENERAL_PATTERN(Multiply, paddle::dialect::MultiplyOp)
DEFINE_GENERAL_PATTERN(Divide, paddle::dialect::DivideOp)
DEFINE_GENERAL_PATTERN(Abs, paddle::dialect::AbsOp)
DEFINE_GENERAL_PATTERN(Full, paddle::dialect::FullOp)
DEFINE_GENERAL_PATTERN(Scale, paddle::dialect::ScaleOp)
DEFINE_GENERAL_PATTERN(BatchNorm, paddle::dialect::BatchNormOp)
DEFINE_GENERAL_PATTERN(BatchNorm_, paddle::dialect::BatchNorm_Op)
DEFINE_GENERAL_PATTERN(Cast, paddle::dialect::CastOp)
DEFINE_GENERAL_PATTERN(Concat, paddle::dialect::ConcatOp)
DEFINE_GENERAL_PATTERN(Conv2d, paddle::dialect::Conv2dOp)
DEFINE_GENERAL_PATTERN(Conv2dTranspose, paddle::dialect::Conv2dTransposeOp)
DEFINE_GENERAL_PATTERN(DepthwiseConv2d, paddle::dialect::DepthwiseConv2dOp)
DEFINE_GENERAL_PATTERN(FullIntArray, paddle::dialect::FullIntArrayOp)
DEFINE_GENERAL_PATTERN(FullLike, paddle::dialect::FullLikeOp)
DEFINE_GENERAL_PATTERN(Hardsigmoid, paddle::dialect::HardsigmoidOp)
DEFINE_GENERAL_PATTERN(Hardswish, paddle::dialect::HardswishOp)
DEFINE_GENERAL_PATTERN(Isnan, paddle::dialect::IsnanOp)
DEFINE_GENERAL_PATTERN(NearestInterp, paddle::dialect::NearestInterpOp)
DEFINE_GENERAL_PATTERN(Pool2d, paddle::dialect::Pool2dOp)
DEFINE_GENERAL_PATTERN(Relu, paddle::dialect::ReluOp)
DEFINE_GENERAL_PATTERN(Relu_, paddle::dialect::Relu_Op)
DEFINE_GENERAL_PATTERN(Reshape, paddle::dialect::ReshapeOp)
DEFINE_GENERAL_PATTERN(Sigmoid, paddle::dialect::SigmoidOp)
DEFINE_GENERAL_PATTERN(Sqrt, paddle::dialect::SqrtOp)
DEFINE_GENERAL_PATTERN(Where, paddle::dialect::WhereOp)

class GcuOpMarkerPass : public pir::PatternRewritePass {
 public:
  GcuOpMarkerPass() : pir::PatternRewritePass("gcu_op_marker_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

#define ADD_PATTERN(OpName) \
  ps.Add(std::make_unique<OpName##OpPattern>(context));
    // Special builtin ops
    ADD_PATTERN(Combine)
    ADD_PATTERN(Split)
    // ADD_PATTERN(Parameter)
    // ADD_PATTERN(Constant)
    // ADD_PATTERN(ConstantTensor)

    // Normal ops
    ADD_PATTERN(Matmul)
    ADD_PATTERN(Add)
    ADD_PATTERN(Add_)
    ADD_PATTERN(Subtract)
    ADD_PATTERN(Multiply)
    ADD_PATTERN(Divide)
    ADD_PATTERN(Abs)
    ADD_PATTERN(Full)
    ADD_PATTERN(Scale)
    ADD_PATTERN(BatchNorm)
    ADD_PATTERN(BatchNorm_)
    ADD_PATTERN(Cast)
    ADD_PATTERN(Concat)
    ADD_PATTERN(Conv2d)
    ADD_PATTERN(Conv2dTranspose)
    ADD_PATTERN(DepthwiseConv2d)
    ADD_PATTERN(FullIntArray)
    ADD_PATTERN(FullLike)
    ADD_PATTERN(Hardsigmoid)
    ADD_PATTERN(Hardswish)
    ADD_PATTERN(Isnan)
    ADD_PATTERN(NearestInterp)
    ADD_PATTERN(Pool2d)
    ADD_PATTERN(Relu)
    ADD_PATTERN(Relu_)
    ADD_PATTERN(Reshape)
    ADD_PATTERN(Sigmoid)
    ADD_PATTERN(Sqrt)
    ADD_PATTERN(Where)

#undef ADD_PATTERN

    return ps;
  }
};
}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateGcuOpMarkerPass() {
  return std::make_unique<GcuOpMarkerPass>();
}
}  // namespace pir

REGISTER_IR_PASS(gcu_op_marker_pass, GcuOpMarkerPass);
