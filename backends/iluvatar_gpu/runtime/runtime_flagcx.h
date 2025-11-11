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
#pragma once
#include <flagcx.h>

#include "paddle/phi/backends/device_ext.h"

extern flagcxHandlerGroup_t flagcx_handler;

C_Status XcclFlagcxGetUniqueIdSize(size_t *size);

C_Status XcclFlagcxGetUniqueId(C_CCLRootId *unique_id);

C_Status XcclFlagcxCommInitRank(size_t nranks,
                                C_CCLRootId *unique_id,
                                size_t rank,
                                C_CCLComm *comm);

C_Status XcclFlagcxDestroyComm(C_CCLComm comm);

C_Status XcclFlagcxAllReduce(void *send_buf,
                             void *recv_buf,
                             size_t count,
                             C_DataType data_type,
                             C_CCLReduceOp op,
                             C_CCLComm comm,
                             C_Stream stream);

C_Status XcclFlagcxBroadcast(void *buf,
                             size_t count,
                             C_DataType data_type,
                             size_t root,
                             C_CCLComm comm,
                             C_Stream stream);

C_Status XcclFlagcxReduce(void *send_buf,
                          void *recv_buf,
                          size_t count,
                          C_DataType data_type,
                          C_CCLReduceOp op,
                          size_t root,
                          C_CCLComm comm,
                          C_Stream stream);

C_Status XcclFlagcxAllGather(void *send_buf,
                             void *recv_buf,
                             size_t count,
                             C_DataType data_type,
                             C_CCLComm comm,
                             C_Stream stream);

C_Status XcclFlagcxReduceScatter(void *send_buf,
                                 void *recv_buf,
                                 size_t count,
                                 C_DataType data_type,
                                 C_CCLReduceOp op,
                                 C_CCLComm comm,
                                 C_Stream stream);

C_Status XcclFlagcxGroupStart();

C_Status XcclFlagcxGroupEnd();

C_Status XcclFlagcxSend(void *send_buf,
                        size_t count,
                        C_DataType data_type,
                        size_t dest_rank,
                        C_CCLComm comm,
                        C_Stream stream);

C_Status XcclFlagcxRecv(void *recv_buf,
                        size_t count,
                        C_DataType data_type,
                        size_t src_rank,
                        C_CCLComm comm,
                        C_Stream stream);

C_Status XcclFlagcxAllToAll(const void **send_buf,
                            const size_t *send_count,
                            const C_DataType *send_dtype,
                            void **recv_buf,
                            const size_t *recv_count,
                            const C_DataType *recv_dtype,
                            size_t rank,
                            size_t nranks,
                            C_CCLComm comm,
                            C_Stream stream);
