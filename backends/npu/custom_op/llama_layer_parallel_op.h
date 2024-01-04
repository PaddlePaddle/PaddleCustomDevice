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

#pragma once
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include "atb/atb_infer.h"
#include "atb_layer_base.h"
#include "paddle/phi/extension.h"
#include "paddle/utils/blank.h"
#include "paddle/utils/variant.h"
#include "paddle/phi/backends/c_comm_lib.h"
namespace phi {
namespace detail {
ccl::CCLComm GetCCLComm(const Place& place, int global_gid);
}
}

#define ATB_FLASH_ATTENTION_MAX_SEQ_LEN 1024

class PpAtbLlamaDecoderLayerParallelOp : public PpAscendAtbOpBase {
public:
  PpAtbLlamaDecoderLayerParallelOp(const std::string &modelName, int32_t layerNum, int32_t batchSize, int maxBatchSize, const phi::CustomContext &dev_ctx);
  ~PpAtbLlamaDecoderLayerParallelOp();
  phi::DenseTensor layerIdTensor_;
  phi::DenseTensor q_seq_len_tensor_;
  phi::DenseTensor token_offset_tensor_;

  void UpdateInputTensorAndParam(const paddle::Tensor &kv_seq_len);
  bool BatchSizeChanged(int32_t batchSize);

private:
  void BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                        std::vector<const phi::DenseTensor *> &outTensors);
  void BindHostTensorForUpdateParam(atb::VariantPack &variantPack);
  atb::Tensor CreateBatchStatusAtbHostTensor();

private:
  std::vector<int32_t> kv_seq_len_param_;
  std::vector<int32_t> q_seq_len_param_;
  std::vector<int32_t> batch_status_param_;

  int32_t layerNum_ = 0;
  int32_t curBatchSize_ = 0;
  int32_t maxBatchSize_ = 0;
};

class PpAtbLlamaEncoderLayerParallelOp : public PpAscendAtbOpBase {
public:
  PpAtbLlamaEncoderLayerParallelOp(const std::string &modelName, int32_t layerNum, int32_t batchSize, int maxBatchSize);
  ~PpAtbLlamaEncoderLayerParallelOp();
  phi::DenseTensor layerIdTensor_;

  void UpdateInputTensorAndParam(const paddle::Tensor &kv_seq_len);

private:
  void BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                        std::vector<const phi::DenseTensor *> &outTensors);
  void BindHostTensorForUpdateParam(atb::VariantPack &variantPack);
  atb::Tensor CreateBatchStatusAtbHostTensor();

private:
  std::vector<int32_t> kv_seq_len_param_;
  std::vector<int32_t> q_seq_len_param_;
  std::vector<int32_t> batch_status_param_;

  int32_t layerNum_ = 0;
  int32_t curBatchSize_ = 0;
  int32_t maxBatchSize_ = 0;
};

class PpAtbLlamaBlockAttnLayerParallelOp : public PpAscendAtbOpBase {
public:
  PpAtbLlamaBlockAttnLayerParallelOp(const std::string &modelName, int layer_num);
  ~PpAtbLlamaBlockAttnLayerParallelOp();
  phi::DenseTensor token_offset_tensor_;
  void UpdateInputTensorAndParam(const paddle::Tensor &block_tables, const paddle::Tensor &seq_len, int32_t block_size);

private:
  void BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                        std::vector<const phi::DenseTensor *> &outTensors);
  void BindHostTensorForUpdateParam(atb::VariantPack &variantPack);
  atb::Tensor CreateBatchStatusAtbHostTensor();
  std::vector<int32_t> batch_status_param_;

private:
  std::vector<int32_t> seq_len_param_;

  int32_t layerNum_ = 0;
  int32_t curBatchSize_ = 0;
  int32_t maxBatchSize_ = 0;
};

#endif