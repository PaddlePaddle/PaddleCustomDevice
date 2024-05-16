// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/funcs/tops_op_utils.h"
#include "kernels/topsflame/include/topsop/topsop_define.h"
#include "kernels/topsflame/include/topsop/topsop_ops.h"

namespace custom_kernel {
namespace {  // NOLINT
template <typename T>
struct topsop_variable {
  explicit topsop_variable(const T& var) { value = var; }

  T value;
};

template <>
struct topsop_variable<phi::DenseTensor> {
  explicit topsop_variable(const phi::DenseTensor& tensor) {
    value = CreateTopsopTensorHandle(tensor);
  }
  ~topsop_variable() {
    if (value != nullptr) {
      topsopDestroyTensor(value);
    }
  }

  topsopTensorHandle_t value = nullptr;
};

template <>
struct topsop_variable<paddle::optional<phi::DenseTensor>> {
  explicit topsop_variable(
      const paddle::optional<phi::DenseTensor>& opt_tensor) {
    value = OptionalTensorToTopsopTensorHandle(opt_tensor);
  }
  ~topsop_variable() {
    if (value != nullptr) {
      topsopDestroyTensor(value);
    }
  }

  topsopTensorHandle_t value = nullptr;
};

template <>
struct topsop_variable<phi::Scalar> {
  explicit topsop_variable(const phi::Scalar& scalar) {
    value = ScalarToTopsopScalar(scalar);
  }

  topsopScalar_t value;
};

template <>
struct topsop_variable<paddle::optional<phi::Scalar>> {
  explicit topsop_variable(const paddle::optional<phi::Scalar>& opt_scalar) {
    value = OptionalScalarToTopsopScalar(opt_scalar);
  }

  topsopScalar_t value;
};

template <>
struct topsop_variable<phi::IntArray> {
  explicit topsop_variable(const phi::IntArray& array) {
    value = IntArrayToTopsopSize(array);
  }

  topsopSize_t value;
};

template <>
struct topsop_variable<paddle::optional<phi::IntArray>> {
  explicit topsop_variable(const paddle::optional<phi::IntArray>& opt_array) {
    value = OptionalIntArrayToTopsopSize(opt_array);
  }

  topsopSize_t value;
};

template <>
struct topsop_variable<std::vector<int64_t>> {
  explicit topsop_variable(const std::vector<int64_t>& array) {
    value = IntArrayToTopsopSize(array);
  }

  topsopSize_t value;
};

template <>
struct topsop_variable<phi::DataType> {
  explicit topsop_variable(const phi::DataType& data_type) {
    value = DataTypeToTopsopDataType(data_type);
  }

  topsopDataType_t value;
};

template <>
struct topsop_variable<std::pair<uint64_t, uint64_t>> {
  explicit topsop_variable(const std::pair<uint64_t, uint64_t>& gen_data) {
    value.seed = gen_data.first;
    value.offset = gen_data.second;
  }

  topsopGenerator_t value;
};

#define TOPS_OP_CALL_MAYBE_SYNC(func, ctx)                                    \
  do {                                                                        \
    VLOG(6) << "[AOT_KERNEL] Call topsflame op: "                             \
            << GetOpNameFromCallStatement(std::string(#func));                \
    GCU_AOT_KERNEL_TRACE(GetOpNameFromCallStatement(std::string(#func)));     \
    auto status = (func);                                                     \
    PADDLE_ENFORCE_EQ(                                                        \
        status,                                                               \
        TOPSOP_STATUS_SUCCESS,                                                \
        phi::errors::Fatal("Failed to call tops op, get error: %d", status)); \
    GcuOpMaybeStreamSync(ctx);                                                \
  } while (false)

#define DEFINE_LAUNCH_TOPSOP(topsop)                                       \
  template <typename... Args>                                              \
  auto launch_##topsop(const phi::CustomContext& dev_ctx,                  \
                       phi::DenseTensor& out,                              \
                       const Args&... args) {                              \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());             \
    auto xout = topsop_variable(out);                                      \
    topsopStatus_t status;                                                 \
    {                                                                      \
      GCU_AOT_KERNEL_TRACE(#topsop);                                       \
      status = topsop(xout.value, topsop_variable(args).value..., stream); \
      GcuOpMaybeStreamSync(dev_ctx);                                       \
    }                                                                      \
    return status;                                                         \
  }

#define DEFINE_LAUNCH_TOPSOP_OUT2(topsop)                                    \
  template <typename... Args>                                                \
  auto launch_##topsop(const phi::CustomContext& dev_ctx,                    \
                       phi::DenseTensor& out1,                               \
                       phi::DenseTensor& out2,                               \
                       const Args&... args) {                                \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());               \
    auto xout1 = topsop_variable(out1);                                      \
    auto xout2 = topsop_variable(out2);                                      \
    topsopStatus_t status;                                                   \
    {                                                                        \
      GCU_AOT_KERNEL_TRACE(#topsop);                                         \
      status = topsop(                                                       \
          xout1.value, xout2.value, topsop_variable(args).value..., stream); \
      GcuOpMaybeStreamSync(dev_ctx);                                         \
    }                                                                        \
    return status;                                                           \
  }

#define LAUNCH_TOPSOP(topsop, dev_ctx, topsop_args...)                   \
  {                                                                      \
    auto op_info = [&]() -> std::string {                                \
      return custom_kernel::GetOpInfo(                                   \
          #topsop,                                                       \
          ##topsop_args,                                                 \
          static_cast<topsStream_t>(dev_ctx.stream()));                  \
    };                                                                   \
    VLOG(6) << "[AOT_KERNEL] Start to launch topsflame, " << op_info();  \
    auto status = launch_##topsop(dev_ctx, ##topsop_args);               \
    PADDLE_ENFORCE_EQ(                                                   \
        status,                                                          \
        TOPSOP_STATUS_SUCCESS,                                           \
        phi::errors::Fatal(                                              \
            "Failed to call topsflame op, get error: %d, details: %s",   \
            status,                                                      \
            op_info().c_str()));                                         \
    VLOG(6) << "Launch topsflame op sucessfully, details:" << op_info(); \
  }
}  // namespace

DEFINE_LAUNCH_TOPSOP(topsopBroadcast)
DEFINE_LAUNCH_TOPSOP(topsopConvert)
DEFINE_LAUNCH_TOPSOP(topsopTranspose)

DEFINE_LAUNCH_TOPSOP(topsopDotGeneral)
DEFINE_LAUNCH_TOPSOP(topsopGather)
DEFINE_LAUNCH_TOPSOP(topsopSlice)
DEFINE_LAUNCH_TOPSOP(topsopReduceAvg)
DEFINE_LAUNCH_TOPSOP(topsopScatter)
DEFINE_LAUNCH_TOPSOP(topsopSortEx)

DEFINE_LAUNCH_TOPSOP(topsopPower)
DEFINE_LAUNCH_TOPSOP(topsopSoftmaxForward)
DEFINE_LAUNCH_TOPSOP(topsopLogSoftmaxForward)

}  // namespace custom_kernel
