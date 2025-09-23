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

#include "runtime_flagcx.h"  // NOLINT

#include <errno.h>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"

C_CCLComm globalComm = nullptr;
flagcxHandlerGroup_t flagcx_handler;

namespace phi {

namespace internal {
inline flagcxDataType_t PDDataTypeToFlagcxDataType(C_DataType type) {
  if (type == C_DataType::FLOAT32) {
    return flagcxFloat;
  } else if (type == C_DataType::BFLOAT16) {
    return flagcxBfloat16;
  } else if (type == C_DataType::UINT8) {
    return flagcxUint8;
  } else if (type == C_DataType::UINT32) {
    return flagcxUint32;
  } else if (type == C_DataType::UINT64) {
    return flagcxUint64;
  } else if (type == C_DataType::INT8) {
    return flagcxInt8;
  } else if (type == C_DataType::INT32) {
    return flagcxInt32;
  } else if (type == C_DataType::INT64) {
    return flagcxInt64;
  } else if (type == C_DataType::FLOAT16) {
    return flagcxHalf;
  } else {
    LOG(ERROR) << "Datatype " << type << " in flagcx is not supported.";
  }
  return flagcxFloat;
}

#define FLAGCX_CHECK(cmd)                                               \
  do {                                                                  \
    flagcxResult_t r = cmd;                                             \
    if (r != flagcxSuccess) {                                           \
      PADDLE_THROW(                                                     \
          common::errors::External("Failed, FLAGCX error %s:%d '%s'\n", \
                                   __FILE__,                            \
                                   __LINE__,                            \
                                   flagcxGetErrorString(r)));           \
    }                                                                   \
  } while (0)
}  // namespace internal
}  // namespace phi

flagcxRedOp_t PDReduceOpToFlagcxReduceOp(C_CCLReduceOp op) {
  if (op == C_CCLReduceOp::MIN) {
    return flagcxMin;
  } else if (op == C_CCLReduceOp::MAX) {
    return flagcxMax;
  } else if (op == C_CCLReduceOp::SUM) {
    return flagcxSum;
  } else if (op == C_CCLReduceOp::PRODUCT) {
    return flagcxProd;
  } else if (op == C_CCLReduceOp::AVG) {
    return flagcxAvg;
  } else {
    LOG(ERROR) << "Reduceop " << op << " in flagcx is not supported.";
  }
}

C_Status XcclFlagcxGetUniqueIdSize(size_t *size) {
  *size = sizeof(flagcxUniqueId);
  return C_SUCCESS;
}

C_Status XcclFlagcxGetUniqueId(C_CCLRootId *unique_id) {
  if (unique_id->sz != sizeof(flagcxUniqueId)) {
    LOG(ERROR) << "unique_id->sz must be equal sizeof(ncclUniqueId)";
    return C_FAILED;
  }
  flagcxUniqueId_t flagcxId =
      reinterpret_cast<flagcxUniqueId *>(unique_id->data);
  FLAGCX_CHECK(flagcxGetUniqueId(&flagcxId));
  unique_id->data = flagcxId;
  return C_SUCCESS;
}

C_Status XcclFlagcxCommInitRank(size_t nranks,
                                C_CCLRootId *unique_id,
                                size_t rank,
                                C_CCLComm *comm) {
  FLAGCX_CHECK(
      flagcxCommInitRank(reinterpret_cast<flagcxComm_t *>(comm),
                         nranks,
                         reinterpret_cast<flagcxUniqueId *>(unique_id->data),
                         rank));
  globalComm = *comm;
  VLOG(4) << "[FLAGCX] comm inited: " << reinterpret_cast<flagcxComm_t>(*comm);
  return C_SUCCESS;
}

C_Status XcclFlagcxDestroyComm(C_CCLComm comm) {
  FLAGCX_CHECK(flagcxCommDestroy(reinterpret_cast<flagcxComm_t>(comm)));
  globalComm = nullptr;
  return C_SUCCESS;
}

C_Status XcclFlagcxAllReduce(void *send_buf,
                             void *recv_buf,
                             size_t count,
                             C_DataType data_type,
                             C_CCLReduceOp op,
                             C_CCLComm comm,
                             C_Stream stream) {
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);

  FLAGCX_CHECK(
      flagcxAllReduce(send_buf,
                      recv_buf,
                      count,
                      phi::internal::PDDataTypeToFlagcxDataType(data_type),
                      PDReduceOpToFlagcxReduceOp(op),
                      reinterpret_cast<flagcxComm_t>(comm),
                      reinterpret_cast<flagcxStream_t>(&cudaStream)));
  return C_SUCCESS;
}

C_Status XcclFlagcxBroadcast(void *buf,
                             size_t count,
                             C_DataType data_type,
                             size_t root,
                             C_CCLComm comm,
                             C_Stream stream) {
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
  FLAGCX_CHECK(
      flagcxBroadcast(static_cast<const void *>(buf),
                      buf,
                      count,
                      phi::internal::PDDataTypeToFlagcxDataType(data_type),
                      root,
                      reinterpret_cast<flagcxComm_t>(comm),
                      reinterpret_cast<flagcxStream_t>(&cudaStream)));
  return C_SUCCESS;
}

C_Status XcclFlagcxReduce(void *send_buf,
                          void *recv_buf,
                          size_t count,
                          C_DataType data_type,
                          C_CCLReduceOp op,
                          size_t root,
                          C_CCLComm comm,
                          C_Stream stream) {
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
  FLAGCX_CHECK(
      flagcxReduce(send_buf,
                   recv_buf,
                   count,
                   phi::internal::PDDataTypeToFlagcxDataType(data_type),
                   PDReduceOpToFlagcxReduceOp(op),
                   root,
                   reinterpret_cast<flagcxComm_t>(comm),
                   reinterpret_cast<flagcxStream_t>(&cudaStream)));
  return C_SUCCESS;
}

C_Status XcclFlagcxAllGather(void *send_buf,
                             void *recv_buf,
                             size_t count,
                             C_DataType data_type,
                             C_CCLComm comm,
                             C_Stream stream) {
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
  FLAGCX_CHECK(
      flagcxAllGather(send_buf,
                      recv_buf,
                      count,
                      phi::internal::PDDataTypeToFlagcxDataType(data_type),
                      reinterpret_cast<flagcxComm_t>(comm),
                      reinterpret_cast<flagcxStream_t>(&cudaStream)));
  return C_SUCCESS;
}

C_Status XcclFlagcxReduceScatter(void *send_buf,
                                 void *recv_buf,
                                 size_t count,
                                 C_DataType data_type,
                                 C_CCLReduceOp op,
                                 C_CCLComm comm,
                                 C_Stream stream) {
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
  FLAGCX_CHECK(
      flagcxReduceScatter(send_buf,
                          recv_buf,
                          count,
                          phi::internal::PDDataTypeToFlagcxDataType(data_type),
                          PDReduceOpToFlagcxReduceOp(op),
                          reinterpret_cast<flagcxComm_t>(comm),
                          reinterpret_cast<flagcxStream_t>(&cudaStream)));
  return C_SUCCESS;
}

C_Status XcclFlagcxGroupStart() {
  FLAGCX_CHECK(flagcxGroupStart(reinterpret_cast<flagcxComm_t>(globalComm)));
  return C_SUCCESS;
}

C_Status XcclFlagcxGroupEnd() {
  FLAGCX_CHECK(flagcxGroupEnd(reinterpret_cast<flagcxComm_t>(globalComm)));
  return C_SUCCESS;
}

C_Status XcclFlagcxSend(void *send_buf,
                        size_t count,
                        C_DataType data_type,
                        size_t dest_rank,
                        C_CCLComm comm,
                        C_Stream stream) {
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
  FLAGCX_CHECK(flagcxSend(send_buf,
                          count,
                          phi::internal::PDDataTypeToFlagcxDataType(data_type),
                          dest_rank,
                          reinterpret_cast<flagcxComm_t>(comm),
                          reinterpret_cast<flagcxStream_t>(&cudaStream)));
  flagcx_handler->devHandle->streamSynchronize(
      reinterpret_cast<flagcxStream_t>(&cudaStream));
  return C_SUCCESS;
}

C_Status XcclFlagcxRecv(void *recv_buf,
                        size_t count,
                        C_DataType data_type,
                        size_t src_rank,
                        C_CCLComm comm,
                        C_Stream stream) {
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
  FLAGCX_CHECK(flagcxRecv(recv_buf,
                          count,
                          phi::internal::PDDataTypeToFlagcxDataType(data_type),
                          src_rank,
                          reinterpret_cast<flagcxComm_t>(comm),
                          reinterpret_cast<flagcxStream_t>(&cudaStream)));
  flagcx_handler->devHandle->streamSynchronize(
      reinterpret_cast<flagcxStream_t>(&cudaStream));
  return C_SUCCESS;
}

C_Status XcclFlagcxAllToAll(const void **send_buf,
                            const size_t *send_count,
                            const C_DataType *send_dtype,
                            void **recv_buf,
                            const size_t *recv_count,
                            const C_DataType *recv_dtype,
                            size_t rank,
                            size_t nranks,
                            C_CCLComm comm,
                            C_Stream stream) {
  flagcxComm_t flagcxComm = reinterpret_cast<flagcxComm_t>(comm);
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
  FLAGCX_CHECK(flagcxGroupStart(flagcxComm));
  for (size_t i = 0; i < nranks; i++) {
    if (send_count[i] > 0) {
      FLAGCX_CHECK(
          flagcxSend(const_cast<void *>(send_buf[i]),
                     send_count[i],
                     phi::internal::PDDataTypeToFlagcxDataType(send_dtype[i]),
                     i,
                     flagcxComm,
                     reinterpret_cast<flagcxStream_t>(&cudaStream)));
    }
    if (recv_count[i] > 0) {
      FLAGCX_CHECK(
          flagcxRecv(const_cast<void *>(recv_buf[i]),
                     recv_count[i],
                     phi::internal::PDDataTypeToFlagcxDataType(recv_dtype[i]),
                     i,
                     flagcxComm,
                     reinterpret_cast<flagcxStream_t>(&cudaStream)));
    }
  }
  FLAGCX_CHECK(flagcxGroupEnd(flagcxComm));
  flagcx_handler->devHandle->streamSynchronize(
      reinterpret_cast<flagcxStream_t>(&cudaStream));
  return C_SUCCESS;
}
