// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include <acl/acl.h>
#include "atb_layer_base.h"
#include "kernels/funcs/format_utils.h"

std::shared_ptr<phi::DenseTensor> PpAscendAtbOpBase::output_ = std::make_shared<phi::DenseTensor>();

void PpAscendAtbOpBase::BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                                              std::vector<const phi::DenseTensor *> &outTensors)
{
  variantPacks_.inTensors.resize(inTensors.size());
  for (size_t i = 0; i < inTensors.size(); i++) {
    variantPacks_.inTensors.at(i) = ConvertDenseTensorToAtbTensor(*(inTensors.at(i)));
    if (variantPacks_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }

  variantPacks_.outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); i++) {
    variantPacks_.outTensors.at(i) = ConvertDenseTensorToAtbTensor(*(outTensors.at(i)));
    if (variantPacks_.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.outTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }
}

void PpAscendAtbOpBase::SetWorkspace(uint64_t workspace_size, const phi::CustomContext * ctx)
{
  if (workspace_size <= g_workspaceSize) {
    return;
  }
  
  g_workspace->Resize(phi::make_ddim({workspace_size}));
  ctx->Alloc(g_workspace, paddle::DataType::INT8);  

  g_workspaceSize = workspace_size;
}

atb::Status PpAscendAtbOpBase::Execute(aclrtStream stream,
                                     std::vector<const phi::DenseTensor *> &inTensors,
                                     std::vector<const phi::DenseTensor *> &outTensors,
                                     const phi::CustomContext * ctx)
{
  uint64_t workspace_size;
  BuildVariantPack(inTensors, outTensors);

  if(context_ == nullptr) {
    atb::CreateContext(&context_);
    context_->SetExecuteStream(stream);
  }

  atb::Status st = operation_->Setup(variantPacks_, workspace_size, context_);
  PADDLE_ENFORCE_EQ(st,
                    0,
                    phi::errors::External("Atb Layer %s Op Setup failed,"
                                          "ret message: %d .", opName_, st));

  if (workspace_size >= 0) {
    if (workspace_size < 512) { // 防止频繁申请释放内存，看情况调整
        workspace_size = 512;
    }
    SetWorkspace(workspace_size, ctx);
  }

  st = operation_->Execute(variantPacks_, (uint8_t *)g_workspace->data(), workspace_size, context_);

  return st;
}

atb::Status PpAscendAtbOpBase::Execute(aclrtStream stream,
                                     std::vector<const phi::DenseTensor *> &inTensors,
                                     std::vector<const phi::DenseTensor *> &outTensors,
                                     const phi::CustomContext * ctx,
                                     int layerid)
{
  uint64_t workspace_size;
  BuildVariantPack(inTensors, outTensors);

  if(context_ == nullptr) {
    atb::CreateContext(&context_);
    context_->SetExecuteStream(stream);
  }
  atb::Status st = operations_.at(layerid)->Setup(variantPacks_, workspace_size, context_);
  PADDLE_ENFORCE_EQ(st,
                    0,
                    phi::errors::External("Atb Layer %s Op Setup failed,"
                                          "ret message: %d .", opName_, st));

  if (workspace_size > 0) {
    if (workspace_size < 512) { // 防止频繁申请释放内存，看情况调整
        workspace_size = 512;
    }
    SetWorkspace(workspace_size, ctx);
  }

  st = operations_.at(layerid)->Execute(variantPacks_, (uint8_t *)g_workspace->data(), workspace_size, context_);

  return st;
}

PpAscendAtbOpBase::PpAscendAtbOpBase(const std::string &opName)
{
  opName_ = opName;
}

PpAscendAtbOpBase::~PpAscendAtbOpBase() {}
#endif
