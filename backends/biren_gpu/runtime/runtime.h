#pragma once

#include <cstddef>
#include <cstdio>

#include "paddle/phi/extension.h"

#define BR_LOG_ERR

#ifdef BR_LOG_ERR
#define LOG_BRS_INF(...)                               \
  {                                                    \
    printf("Line-%s @ File-%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__);                               \
  }
#else
#define LOG_BRS_INF(...)
#endif

#define PARAM_CHECK_PTR(ptr, err_id)     \
  {                                      \
    if (NULL == ptr) {                   \
      LOG_BRS_INF("input ptr error! \n") \
      return err_id;                     \
    }                                    \
  }

#define PARAM_CHECK_BOOL(flag, err_id)   \
  {                                      \
    if (false == flag) {                 \
      LOG_BRS_INF("input ptr error! \n") \
      return err_id;                     \
    }                                    \
  }

#define PARAM_CHECK_MEM_SIZE(size, size_max, err_id) \
  {                                                  \
    if (size > size_max) {                           \
      LOG_BRS_INF("memory over size! \n")            \
      return err_id;                                 \
    }                                                \
  }

C_Status memcpy_h2d(const C_Device device, void *dst, const void *src,
                    size_t size);

C_Status memcpy_d2d(const C_Device device, void *dst, const void *src,
                    size_t size);

C_Status memcpy_d2h(const C_Device device, void *dst, const void *src,
                    size_t size);

C_Status async_memcpy_h2d(const C_Device device, C_Stream stream, void *dst,
                          const void *src, size_t size);

C_Status async_memcpy_d2d(const C_Device device, C_Stream stream, void *dst,
                          const void *src, size_t size);

C_Status async_memcpy_d2h(const C_Device device, C_Stream stream, void *dst,
                          const void *src, size_t size);
