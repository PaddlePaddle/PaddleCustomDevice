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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class GcuConv2dAddFusePattern1 : public paddle::drr::DrrPatternBase {
 private:
  std::string conv2d_op_name_;

 public:
  std::string name() const override { return "GcuConv2dAddFusePattern1"; }

  explicit GcuConv2dAddFusePattern1(const std::string &conv2d_op_name)
      : conv2d_op_name_(conv2d_op_name) {}
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &conv2d =
        pat.Op(conv2d_op_name_,
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &act_op = pat.Op("pd_op.leaky_relu");
    conv2d({&pat.Tensor("input"), &pat.Tensor("filter")},
           {&pat.Tensor("conv2d_out")});
    pat.Tensor("add_out") = add(pat.Tensor("conv2d_out"), pat.Tensor("bias"));
    pat.Tensor("act_out") = act_op(pat.Tensor("add_out"));
    pat.AddConstraint([this](
                          const paddle::drr::MatchContext &match_ctx) -> bool {
      auto bias_shape = pir::GetShapeFromValue(match_ctx.Tensor("bias"));
      auto add_input_shape =
          pir::GetShapeFromValue(match_ctx.Tensor("conv2d_out"));
      auto bias_ = 1;
      auto add_input_ = 1;
      for (auto &bias_dim : bias_shape) {
        bias_ *= bias_dim;
      }
      for (auto &add_input_dim : add_input_shape) {
        add_input_ *= add_input_dim;
      }
      if (bias_ == add_input_) {
        return false;
      }

      if (!pir::ValueIsPersistable(match_ctx.Tensor("bias"))) {
        return false;
      }

      auto padding_algorithm = match_ctx.Attr<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT" && padding_algorithm != "SAME" &&
          padding_algorithm != "VALID") {
        return false;
      }

      auto groups = match_ctx.Attr<int>("groups");
      if (groups < 1) {
        return false;
      }

      auto data_format = match_ctx.Attr<std::string>("data_format");
      if (data_format != "NCHW" && data_format != "AnyLayout" &&
          data_format != "NHWC") {
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_conv2d_add_act =
        res.Op(paddle::dialect::FusedConv2dAddActOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"activation", res.StrAttr("leaky_relu")},
                   {"split_channels", res.VectorInt32Attr({})},
                   {"exhaustive_search", res.BoolAttr(false)},
                   {"workspace_size_MB", res.Int32Attr(32)},
                   {"fuse_alpha", res.Float32Attr(0.0f)},
               }});

    fused_conv2d_add_act({&res.Tensor("input"),
                          &res.Tensor("filter"),
                          &res.Tensor("bias"),
                          &res.InputNoneTensor()},
                         {&res.Tensor("act_out"), &res.OutputNoneTensor()});
  }
};

class GcuConv2dAddFusePass1 : public pir::PatternRewritePass {
 public:
  GcuConv2dAddFusePass1()
      : pir::PatternRewritePass("gcu_conv2d_add_fuse_pass_1", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<GcuConv2dAddFusePattern1>(
        context, paddle::dialect::Conv2dOp::name()));
    ps.Add(paddle::drr::Create<GcuConv2dAddFusePattern1>(
        context, paddle::dialect::DepthwiseConv2dOp::name()));
    return ps;
  }
};

}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateGcuConv2dAddFusePass1() {
  return std::make_unique<GcuConv2dAddFusePass1>();
}
}  // namespace pir

REGISTER_IR_PASS(gcu_conv2d_add_fuse_pass_1, GcuConv2dAddFusePass1);
