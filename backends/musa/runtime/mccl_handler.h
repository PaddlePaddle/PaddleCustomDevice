// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
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
#include <sstream>
#include <unordered_map>

#include "utils.h"  // NOLINT
namespace musa {

namespace mccl {

static mcclRedOp_t PDReduceOp2McclReduceOp(C_CCLReduceOp op) {
  static std::unordered_map<C_CCLReduceOp, mcclRedOp_t> pd_op_to_mccl_op{
      {C_CCLReduceOp::MIN, mcclMin},
      {C_CCLReduceOp::MAX, mcclMax},
      {C_CCLReduceOp::SUM, mcclSum},
      {C_CCLReduceOp::PRODUCT, mcclProd},
      {C_CCLReduceOp::AVG, mcclAvg}};

  auto iter = pd_op_to_mccl_op.find(op);
  if (iter == pd_op_to_mccl_op.end()) {
    std::stringstream ss;
    ss << "Reduceop " << op << " in mccl is not supported.";
    LOG(ERROR) << ss.str();
    PD_CHECK(false, ss.str().c_str());
  }

  return iter->second;
}

static mcclDataType_t PDDataType2McclDataType(C_DataType type) {
  static std::unordered_map<C_DataType, mcclDataType_t> pd_type_to_mccl_type{
      {C_DataType::UINT8, mcclUint8},
      {C_DataType::UINT32, mcclUint32},
      {C_DataType::UINT64, mcclUint64},
      {C_DataType::UINT8, mcclUint8},
      {C_DataType::INT8, mcclInt8},
      {C_DataType::INT32, mcclInt32},
      {C_DataType::INT64, mcclInt64},
      {C_DataType::FLOAT16, mcclFloat16},
      {C_DataType::FLOAT32, mcclFloat32},
      {C_DataType::FLOAT64, mcclFloat64}
      // {C_DataType::BFLOAT16, mcclBfloat16}, TODO(jihong.zhong)： fix it
  };

  auto iter = pd_type_to_mccl_type.find(type);
  if (iter == pd_type_to_mccl_type.end()) {
    std::stringstream ss;
    ss << "Datatype " << type << " in mccl is not supported.";
    LOG(ERROR) << ss.str();
    PD_CHECK(false, ss.str().c_str());
  }

  return iter->second;
}

C_Status McclGroupStart();
C_Status McclGroupEnd();

C_Status McclGetUniqueIdSize(size_t *size);
C_Status McclGetUniqueId(C_CCLRootId *unique_id);
C_Status McclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm);
C_Status McclDestroyComm(C_CCLComm comm);
C_Status McclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream);
C_Status McclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream);
C_Status McclReduce(void *send_buf,
                    void *recv_buf,
                    size_t count,
                    C_DataType data_type,
                    C_CCLReduceOp op,
                    size_t root,
                    C_CCLComm comm,
                    C_Stream stream);
C_Status McclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream);
C_Status McclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream);
C_Status McclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream);
C_Status McclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream);
C_Status McclAll2All(const void **send_buf,
                     const size_t *send_count,
                     const C_DataType *send_dtype,
                     void **recv_buf,
                     const size_t *recv_count,
                     const C_DataType *recv_dtype,
                     size_t rank,
                     size_t nranks,
                     C_CCLComm comm,
                     C_Stream stream);
}  // namespace mccl

}  // namespace musa
