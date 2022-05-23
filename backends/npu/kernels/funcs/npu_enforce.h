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

#include <string>

#include <boost/variant.hpp>

#include "paddle/phi/core/enforce.h"

#include "acl/acl.h"
#include "hccl/hccl_types.h"

namespace custom_kernel {

namespace details {
template <typename T>
struct NPUStatusType {};

#define DEFINE_NPU_STATUS_TYPE(type, success_value) \
  template <>                                       \
  struct NPUStatusType<type> {                      \
    using Type = type;                              \
    static constexpr Type kSuccess = success_value; \
  }

DEFINE_NPU_STATUS_TYPE(aclError, ACL_ERROR_NONE);
DEFINE_NPU_STATUS_TYPE(HcclResult, HCCL_SUCCESS);

#define DEFINE_SAFE_BOOST_GET(                                               \
    __InputType, __OutputType, __OutputTypePtr, __FuncName)                  \
  template <typename OutputType, typename InputType>                         \
  auto __FuncName(                                                           \
      __InputType input, const char* expression, const char* file, int line) \
      ->typename std::conditional<std::is_pointer<InputType>::value,         \
                                  __OutputTypePtr,                           \
                                  __OutputType>::type {                      \
    try {                                                                    \
      return boost::get<OutputType>(input);                                  \
    } catch (boost::bad_get&) {                                              \
      HANDLE_THE_ERROR                                                       \
      throw ::phi::enforce::EnforceNotMet(                                   \
          phi::errors::InvalidArgument(                                      \
              "boost::get failed, cannot get value "                         \
              "(%s) by type %s, its type is %s.",                            \
              expression,                                                    \
              phi::enforce::demangle(typeid(OutputType).name()),             \
              phi::enforce::demangle(input.type().name())),                  \
          file,                                                              \
          line);                                                             \
      END_HANDLE_THE_ERROR                                                   \
    }                                                                        \
  }

DEFINE_SAFE_BOOST_GET(const InputType&,
                      const OutputType&,
                      const OutputType*,
                      SafeBoostGetConst);

}  // namespace details

#define BOOST_GET_CONST(__TYPE, __VALUE)             \
  custom_kernel::details::SafeBoostGetConst<__TYPE>( \
      __VALUE, #__VALUE, __FILE__, __LINE__)

inline std::string build_npu_error_msg(aclError stat) {
  std::ostringstream sout;
  sout << " ACL error, the error code is : " << stat << ". ";
  return sout.str();
}

inline std::string build_npu_error_msg(HcclResult stat) {
  std::ostringstream sout;
  sout << " HCCL error, the error code is : " << stat << ". ";
  return sout.str();
}

#define PADDLE_ENFORCE_NPU_SUCCESS(COND)                                       \
  do {                                                                         \
    auto __cond__ = (COND);                                                    \
    using __NPU_STATUS_TYPE__ = decltype(__cond__);                            \
    constexpr auto __success_type__ = ::custom_kernel::details::NPUStatusType< \
        __NPU_STATUS_TYPE__>::kSuccess;                                        \
    if (UNLIKELY(__cond__ != __success_type__)) {                              \
      auto __summary__ = ::phi::errors::External(                              \
          ::custom_kernel::build_npu_error_msg(__cond__));                     \
      __THROW_ERROR_INTERNAL__(__summary__);                                   \
    }                                                                          \
  } while (0)

}  // namespace custom_kernel
