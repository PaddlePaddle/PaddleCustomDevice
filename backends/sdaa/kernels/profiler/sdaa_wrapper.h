// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#pragma once

#include <chrono>

#include "kernels/profiler/RecordEvent.h"
#include "paddle/phi/core/enforce.h"
#include "runtime/flags.h"
#include "sdcops.h"  // NOLINT

DECLARE_bool(sdaa_error_check);

// NOTE(liaotjanju): When one api is executed sdaa error is checked when
// FLAGS_sdaa_error_check is enabled
#define CHECK_ERROR()                                                        \
  ({                                                                         \
    if (FLAGS_sdaa_error_check) {                                            \
      auto&& sdaa_error = sdaaGetLastError();                                \
      if (sdaaSuccess != sdaa_error && sdaaErrorNotReady != sdaa_error) {    \
        PADDLE_THROW(phi::errors::PreconditionNotMet(                        \
            "line:%d, Error:%s", __LINE__, sdaaGetErrorString(sdaa_error))); \
      }                                                                      \
    }                                                                        \
  })

#define RECORD_FUNCTION_SYNC(SDAA_OP)                             \
  ({                                                              \
    uint64_t cpu_begin = PosixInNsec();                           \
    auto&& sdaa_result = SDAA_OP;                                 \
    uint64_t cpu_end = PosixInNsec();                             \
    auto thread_id = RecordEvent::Instance().getThread();         \
    TraceEvent event(#SDAA_OP, cpu_begin, cpu_end, 0, thread_id); \
    RecordEvent::Instance().init(event);                          \
    sdaa_result;                                                  \
  })

#define RECORD_FUNCTION(SDAA_OP)                                             \
  (RecordEvent::Instance().GetProfilerMode() ? RECORD_FUNCTION_SYNC(SDAA_OP) \
                                             : SDAA_OP)

// NOTE: When SDPTI is enabled, sdaa info is provided by SDPTI
#define RECORD_SDAA_FUNCTION(SDAA_OP)              \
  (RecordEvent::Instance().GetProfilerMode() &&    \
           !RecordEvent::Instance().GetSdptiMode() \
       ? RECORD_FUNCTION_SYNC(SDAA_OP)             \
       : SDAA_OP)

#define RECORD_FUNCTION_SYNC_WITH_MSG(SDAA_OP, MSG)                       \
  ({                                                                      \
    uint64_t cpu_begin = PosixInNsec();                                   \
    auto&& sdaa_result = SDAA_OP;                                         \
    uint64_t cpu_end = PosixInNsec();                                     \
    auto thread_id = RecordEvent::Instance().getThread();                 \
    TraceEvent event(#SDAA_OP, cpu_begin, cpu_end, 0, thread_id, 0, MSG); \
    RecordEvent::Instance().init(event);                                  \
    sdaa_result;                                                          \
  })

#define RECORD_FUNCTION_WITH_MSG(SDAA_OP, MSG)       \
  (RecordEvent::Instance().GetProfilerMode()         \
       ? RECORD_FUNCTION_SYNC_WITH_MSG(SDAA_OP, MSG) \
       : SDAA_OP)

// FIXME(liaotianju): It's strange using PADDLE_ENFORCE leads to some failed
// case, use PADDLE_THROW for now
#define TECODNN_CHECK(SDAA_OP)                                             \
  do {                                                                     \
    auto&& sdaa_result = RECORD_FUNCTION(SDAA_OP);                         \
    if (sdaa_result != TECODNN_STATUS_SUCCESS) {                           \
      PADDLE_THROW(phi::errors::PreconditionNotMet(                        \
          "line: %d,TECODNN failure\nError:%d\n", __LINE__, sdaa_result)); \
    }                                                                      \
    CHECK_ERROR();                                                         \
  } while (0);

#define TBLAS_CHECK(SDAA_OP)                                                \
  do {                                                                      \
    auto&& sdaa_result = RECORD_FUNCTION(SDAA_OP);                          \
    if (sdaa_result != TBLAS_STATUS_SUCCESS) {                              \
      PADDLE_THROW(phi::errors::PreconditionNotMet(                         \
          "line: %d,TECOBLAS failure\nError:%d\n", __LINE__, sdaa_result)); \
    }                                                                       \
    CHECK_ERROR();                                                          \
  } while (0);

#define TBLAS_CHECK_WITH_MSG(SDAA_OP, MSG)                                  \
  do {                                                                      \
    auto&& sdaa_result = RECORD_FUNCTION_WITH_MSG(SDAA_OP, MSG);            \
    if (sdaa_result != TBLAS_STATUS_SUCCESS) {                              \
      PADDLE_THROW(phi::errors::PreconditionNotMet(                         \
          "line: %d,TECOBLAS failure\nError:%d\n", __LINE__, sdaa_result)); \
    }                                                                       \
    CHECK_ERROR();                                                          \
  } while (0);

#define TCUS_CHECK(SDAA_OP)                                             \
  do {                                                                  \
    auto&& sdaa_result = RECORD_FUNCTION(SDAA_OP);                      \
    if (sdaa_result != SDC_SUCCESS) {                                   \
      PADDLE_THROW(phi::errors::PreconditionNotMet(                     \
          "line: %d,TCUS failure\nError:%d\n", __LINE__, sdaa_result)); \
    }                                                                   \
    CHECK_ERROR();                                                      \
  } while (0);

#ifdef checkSdaaErrors
#undef checkSdaaErrors
#endif
#define checkSdaaErrors(SDAA_OP)                                          \
  do {                                                                    \
    auto&& sdaa_result = RECORD_SDAA_FUNCTION(SDAA_OP);                   \
    if (sdaaSuccess != sdaa_result && sdaaErrorNotReady != sdaa_result) { \
      PADDLE_THROW(phi::errors::PreconditionNotMet(                       \
          "Sdaa runtime error in line %d of file %s : %s \n",             \
          __LINE__,                                                       \
          __FILE__,                                                       \
          sdaaGetErrorString(sdaa_result)));                              \
    }                                                                     \
  } while (0);

#define TCCL_CHECK(SDAA_OP)                                             \
  do {                                                                  \
    auto&& sdaa_result = RECORD_FUNCTION(SDAA_OP);                      \
    if (sdaa_result != tcclSuccess) {                                   \
      PADDLE_THROW(phi::errors::PreconditionNotMet(                     \
          "line: %d,TCCL failure\nError:%d\n", __LINE__, sdaa_result)); \
    }                                                                   \
    CHECK_ERROR();                                                      \
  } while (0);

// Get system-wide realtime clock in nanoseconds
inline uint64_t PosixInNsec() {
#ifdef _POSIX_C_SOURCE
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec * 1000 * 1000 * 1000 + tp.tv_nsec;
#else
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
#endif
}
