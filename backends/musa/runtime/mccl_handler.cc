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

#include "mccl_handler.h"  // NOLINT

namespace musa {

namespace mccl {

C_Status McclGetUniqueIdSize(size_t *size) {
  *size = sizeof(mcclUniqueId);
  return C_SUCCESS;
}

C_Status McclGetUniqueId(C_CCLRootId *unique_id) {
  if (unique_id->sz != sizeof(mcclUniqueId)) {
    // LOG(ERROR) << "unique_id->sz must be equal sizeof(mcclUniqueId)";
    return C_FAILED;
  }
  MCCL_CHECK(
      mcclGetUniqueId(reinterpret_cast<mcclUniqueId *>(unique_id->data)));

  return C_SUCCESS;
}

C_Status McclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  MCCL_CHECK(
      mcclCommInitRank(reinterpret_cast<mcclComm_t *>(comm),
                       nranks,
                       *(reinterpret_cast<mcclUniqueId *>(unique_id->data)),
                       rank));
  return C_SUCCESS;
}

C_Status McclDestroyComm(C_CCLComm comm) {
  MCCL_CHECK(mcclCommDestroy(reinterpret_cast<mcclComm_t>(comm)));
  return C_SUCCESS;
}

C_Status McclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  MCCL_CHECK(mcclAllReduce(send_buf,
                           recv_buf,
                           count,
                           PDDataType2McclDataType(data_type),
                           PDReduceOp2McclReduceOp(op),
                           reinterpret_cast<mcclComm_t>(comm),
                           reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status McclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  MCCL_CHECK(mcclBroadcast(static_cast<const void *>(buf),
                           buf,
                           count,
                           PDDataType2McclDataType(data_type),
                           root,
                           reinterpret_cast<mcclComm_t>(comm),
                           reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status McclReduce(void *send_buf,
                    void *recv_buf,
                    size_t count,
                    C_DataType data_type,
                    C_CCLReduceOp op,
                    size_t root,
                    C_CCLComm comm,
                    C_Stream stream) {
  MCCL_CHECK(mcclReduce(send_buf,
                        recv_buf,
                        count,
                        PDDataType2McclDataType(data_type),
                        PDReduceOp2McclReduceOp(op),
                        root,
                        reinterpret_cast<mcclComm_t>(comm),
                        reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status McclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  MCCL_CHECK(mcclAllGather(send_buf,
                           recv_buf,
                           count,
                           PDDataType2McclDataType(data_type),
                           reinterpret_cast<mcclComm_t>(comm),
                           reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status McclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  MCCL_CHECK(mcclReduceScatter(send_buf,
                               recv_buf,
                               count,
                               PDDataType2McclDataType(data_type),
                               PDReduceOp2McclReduceOp(op),
                               reinterpret_cast<mcclComm_t>(comm),
                               reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status McclGroupStart() {
  MCCL_CHECK(mcclGroupStart());
  return C_SUCCESS;
}

C_Status McclGroupEnd() {
  MCCL_CHECK(mcclGroupEnd());
  return C_SUCCESS;
}

C_Status McclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  MCCL_CHECK(mcclSend(send_buf,
                      count,
                      PDDataType2McclDataType(data_type),
                      dest_rank,
                      reinterpret_cast<mcclComm_t>(comm),
                      reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}

C_Status McclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  MCCL_CHECK(mcclRecv(recv_buf,
                      count,
                      PDDataType2McclDataType(data_type),
                      src_rank,
                      reinterpret_cast<mcclComm_t>(comm),
                      reinterpret_cast<musaStream_t>(stream)));
  return C_SUCCESS;
}
C_Status McclAll2All(const void **send_buf,
                     const size_t *send_count,
                     const C_DataType *send_dtype,
                     void **recv_buf,
                     const size_t *recv_count,
                     const C_DataType *recv_dtype,
                     size_t rank,
                     size_t nranks,
                     C_CCLComm comm,
                     C_Stream stream) {
  MCCL_CHECK(mcclGroupStart());
  for (size_t i = 0; i < nranks; ++i) {
    if (send_count[i] != 0)
      MCCL_CHECK(mcclSend(send_buf[i],
                          send_count[i],
                          PDDataType2McclDataType(send_dtype[i]),
                          i,
                          reinterpret_cast<mcclComm_t>(comm),
                          reinterpret_cast<musaStream_t>(stream)));
    if (recv_count[i] != 0)
      MCCL_CHECK(mcclRecv(recv_buf[i],
                          recv_count[i],
                          PDDataType2McclDataType(recv_dtype[i]),
                          i,
                          reinterpret_cast<mcclComm_t>(comm),
                          reinterpret_cast<musaStream_t>(stream)));
  }
  MCCL_CHECK(mcclGroupEnd());
  return C_SUCCESS;
}

}  // namespace mccl

}  // namespace musa
