/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>

#include <any>
#include <bitset>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace custom_pass {
class FusedConv2dAddActAppendPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string act_name_;

 public:
  explicit FusedConv2dAddActAppendPattern(const std::string &act_name)
      : act_name_(act_name) {}
  std::string name() const override { return "FusedConv2dAddActAppendPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &fused_conv2d_add_act =
        pat.Op(paddle::dialect::FusedConv2dAddActOp::name(),
               {{"strides", pat.Attr("strides")},
                {"paddings", pat.Attr("paddings")},
                {"padding_algorithm", pat.Attr("padding_algorithm")},
                {"dilations", pat.Attr("dilations")},
                {"groups", pat.Attr("groups")},
                {"data_format", pat.Attr("data_format")},
                {"activation", pat.Attr("activation")},
                {"split_channels", pat.Attr("split_channels")},
                {"exhaustive_search", pat.Attr("exhaustive_search")},
                {"workspace_size_MB", pat.Attr("workspace_size_MB")},
                {"fuse_alpha", pat.Attr("fuse_alpha")}});
    const auto &act_op = pat.Op("pd_op." + act_name_);
    fused_conv2d_add_act({&pat.Tensor("input"),
                          &pat.Tensor("filter"),
                          &pat.Tensor("bias"),
                          &pat.Tensor("residual_data")},
                         {&pat.Tensor("fused_conv2d_add_act_out"),
                          &pat.Tensor("fused_conv2d_add_act_outputs")});
    pat.Tensor("act_out") = act_op(pat.Tensor("fused_conv2d_add_act_out"));
    pat.AddConstraint([this](
                          const paddle::drr::MatchContext &match_ctx) -> bool {
      auto activation = match_ctx.Attr<std::string>("activation");
      if (activation != "identity") {
        return false;
      }
      auto groups = match_ctx.Attr<int>("groups");
      auto input_shape = pir::GetShapeFromValue(match_ctx.Tensor("input"));
      auto filter_shape = pir::GetShapeFromValue(match_ctx.Tensor("filter"));
      int ic = input_shape[1];
      int oc = filter_shape[0];
      if ((groups == ic) && (ic == oc)) {  // is depthwise_conv
        return false;
      }
      return true;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &fused_conv2d_add_act_op_res =
        res.Op(paddle::dialect::FusedConv2dAddActOp::name(),
               {{
                   {"strides", pat.Attr("strides")},
                   {"paddings", pat.Attr("paddings")},
                   {"padding_algorithm", pat.Attr("padding_algorithm")},
                   {"dilations", pat.Attr("dilations")},
                   {"groups", pat.Attr("groups")},
                   {"data_format", pat.Attr("data_format")},
                   {"activation", res.StrAttr(act_name_)},
                   {"split_channels", pat.Attr("split_channels")},
                   {"exhaustive_search", pat.Attr("exhaustive_search")},
                   {"workspace_size_MB", pat.Attr("workspace_size_MB")},
                   {"fuse_alpha", pat.Attr("fuse_alpha")},
               }});
    fused_conv2d_add_act_op_res(
        {&res.Tensor("input"),
         &res.Tensor("filter"),
         &res.Tensor("bias"),
         &res.Tensor("residual_data")},
        {&res.Tensor("act_out"), &res.Tensor("fused_conv2d_add_act_outputs")});
  }
};

class FusedConv2dAddActAppendPass : public pir::PatternRewritePass {
 public:
  FusedConv2dAddActAppendPass()
      : pir::PatternRewritePass("fused_conv2d_add_act_append_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    const std::unordered_set<std::string> gcu_act_set = {
        "relu", "swish", "sigmoid", "hardswish" /*, "leaky_relu"*/};
    for (auto act_name : gcu_act_set) {
      ps.Add(paddle::drr::Create<FusedConv2dAddActAppendPattern>(context,
                                                                 act_name));
    }
    return ps;
  }
};

}  // namespace custom_pass

REGISTER_IR_PASS(fused_conv2d_add_act_append_pass,
                 custom_pass::FusedConv2dAddActAppendPass);
