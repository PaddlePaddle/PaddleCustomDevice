// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <cncl.h>
#include <cnnl.h>
#include <cnpapi.h>
#include <cnrt.h>
#include <mlu_op.h>

#include "glog/logging.h"
#include "paddle/phi/core/os_info.h"
#include "paddle/phi/extension.h"
#include "runtime/process_cnpapi_data.h"

template <typename T>
struct mluStatusType {};

#define DEFINE_CUSTOM_MLU_STATUS_TYPE(type, success_value) \
  template <>                                              \
  struct mluStatusType<type> {                             \
    using Type = type;                                     \
    static constexpr Type kSuccess = success_value;        \
  }

#define CNPAPI_CALL(call)                                                    \
  do {                                                                       \
    cnpapiResult _status = call;                                             \
    if (_status != CNPAPI_SUCCESS) {                                         \
      const char *errstr;                                                    \
      cnpapiGetResultString(_status, &errstr);                               \
      LOG(ERROR) << "Function " << #call << " failed with error " << errstr; \
    }                                                                        \
  } while (0)

DEFINE_CUSTOM_MLU_STATUS_TYPE(cnrtRet_t, cnrtSuccess);
DEFINE_CUSTOM_MLU_STATUS_TYPE(cnnlStatus_t, CNNL_STATUS_SUCCESS);
DEFINE_CUSTOM_MLU_STATUS_TYPE(mluOpStatus_t, MLUOP_STATUS_SUCCESS);
DEFINE_CUSTOM_MLU_STATUS_TYPE(cnclResult_t, CNCL_RET_SUCCESS);

/*************** CNRT ERROR ***************/
inline bool is_error(cnrtRet_t e) { return e != cnrtSuccess; }

inline std::string build_mlu_error_msg(cnrtRet_t e) {
  std::ostringstream sout;
  sout << "MLU CNRT error(" << e << "), " << cnrtGetErrorName(e) << ": "
       << cnrtGetErrorStr(e);
  return sout.str();
}

/*************** CNNL ERROR ***************/
inline bool is_error(cnnlStatus_t stat) { return stat != CNNL_STATUS_SUCCESS; }

inline std::string build_mlu_error_msg(cnnlStatus_t stat) {
  std::ostringstream sout;
  sout << "MLU CNNL error(" << stat << "), " << cnnlGetErrorString(stat)
       << ". ";
  return sout.str();
}

/*************** MLUOP ERROR ***************/
inline bool is_error(mluOpStatus_t stat) {
  return stat != MLUOP_STATUS_SUCCESS;
}

inline std::string build_mlu_error_msg(mluOpStatus_t stat) {
  std::ostringstream sout;
  sout << "MLU OP error(" << stat << "), " << mluOpGetErrorString(stat) << ". ";
  return sout.str();
}

/*************** CNCL ERROR ***************/
inline bool is_error(cnclResult_t e) { return e != CNCL_RET_SUCCESS; }

inline std::string build_mlu_error_msg(cnclResult_t e) {
  std::ostringstream sout;
  sout << "MLU CNCL error(" << e << "), " << cnclGetErrorStr(e) << ". ";
  return sout.str();
}

#define PADDLE_ENFORCE_MLU_SUCCESS(COND)                \
  do {                                                  \
    auto __cond__ = (COND);                             \
    using __MLU_STATUS_TYPE__ = decltype(__cond__);     \
    constexpr auto __success_type__ =                   \
        mluStatusType<__MLU_STATUS_TYPE__>::kSuccess;   \
    if (UNLIKELY(__cond__ != __success_type__)) {       \
      auto __summary__ = build_mlu_error_msg(__cond__); \
      __THROW_ERROR_INTERNAL__(__summary__);            \
    }                                                   \
  } while (0)

struct mluStream {
  cnnlHandle_t handle;
  mluOpHandle_t op_handle;
  cnrtQueue_t queue;
};
typedef mluStream *mluStream_t;

struct lastCommStream {
  static lastCommStream &Instance() {
    static lastCommStream last_comm_stream;
    return last_comm_stream;
  }

  void Update(cnrtQueue_t q) {
    if (p_queue == nullptr) {
      VLOG(4) << "previous comm queue is nullptr, set to queue: " << q;
      p_queue = q;
    } else if (p_queue != q) {
      VLOG(4) << "use different comm_queue, prev: " << p_queue
              << " current: " << q;
      VLOG(4) << "sync prev_queue: " << p_queue;
      cnrtQueueSync(p_queue);
      p_queue = q;
    }
  }

  cnrtQueue_t get() { return p_queue; }

 private:
  lastCommStream() = default;
  cnrtQueue_t p_queue = nullptr;
};

inline cnnlHandle_t GetHandle(const C_Stream stream) {
  return reinterpret_cast<mluStream_t>(stream)->handle;
}
inline cnnlHandle_t GetHandle(void *stream) {
  return reinterpret_cast<mluStream_t>(stream)->handle;
}

inline mluOpHandle_t GetOpHandle(const C_Stream stream) {
  return reinterpret_cast<mluStream_t>(stream)->op_handle;
}
inline mluOpHandle_t GetOpHandle(void *stream) {
  return reinterpret_cast<mluStream_t>(stream)->op_handle;
}

inline cnrtQueue_t GetQueue(const C_Stream stream) {
  return reinterpret_cast<mluStream_t>(stream)->queue;
}
inline cnrtQueue_t GetQueue(void *stream) {
  return reinterpret_cast<mluStream_t>(stream)->queue;
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
