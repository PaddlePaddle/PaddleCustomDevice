#pragma once

#include <cncl.h>
#include <cnnl.h>
#include <cnpapi.h>
#include <cnrt.h>

#include "glog/logging.h"
#include "paddle/phi/extension.h"

template <typename T>
struct CustomMLUStatusType {};

#define DEFINE_CUSTOM_MLU_STATUS_TYPE(type, success_value) \
  template <>                                              \
  struct CustomMLUStatusType<type> {                       \
    using Type = type;                                     \
    static constexpr Type kSuccess = success_value;        \
  }

DEFINE_CUSTOM_MLU_STATUS_TYPE(cnrtRet_t, cnrtSuccess);
DEFINE_CUSTOM_MLU_STATUS_TYPE(cnnlStatus_t, CNNL_STATUS_SUCCESS);
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

/*************** CNCL ERROR ***************/
inline bool is_error(cnclResult_t e) { return e != CNCL_RET_SUCCESS; }

inline std::string build_mlu_error_msg(cnclResult_t e) {
  std::ostringstream sout;
  sout << "MLU CNCL error(" << e << "), " << cnclGetErrorStr(e) << ". ";
  return sout.str();
}

#define PADDLE_ENFORCE_MLU_SUCCESS(COND)                    \
  do {                                                      \
    auto __cond__ = (COND);                                 \
    using __MLU_STATUS_TYPE__ = decltype(__cond__);         \
    constexpr auto __success_type__ =                       \
        CustomMLUStatusType<__MLU_STATUS_TYPE__>::kSuccess; \
    if (UNLIKELY(__cond__ != __success_type__)) {           \
      auto __summary__ = build_mlu_error_msg(__cond__);     \
      __THROW_ERROR_INTERNAL__(__summary__);                \
    }                                                       \
  } while (0)

struct CustomMLUStream {
  cnnlHandle_t handle;
  cnrtQueue_t queue;
};
typedef CustomMLUStream *mluStream_t;

inline cnnlHandle_t GetHandle(const C_Stream stream) {
  return reinterpret_cast<mluStream_t>(stream)->handle;
}
inline cnnlHandle_t GetHandle(void *stream) {
  return reinterpret_cast<mluStream_t>(stream)->handle;
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
