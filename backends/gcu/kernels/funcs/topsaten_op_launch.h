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
#include "kernels/funcs/topsaten_op_utils.h"
#include "topsaten/topsaten_define.h"
#include "topsaten/topsaten_ops.h"
#include "topsaten/topsaten_vllm.h"

namespace custom_kernel {

namespace {  // NOLINT

template <typename T>
struct topsaten_variable {
  explicit topsaten_variable(const T& var) { value = var; }

  T value;
};

template <>
struct topsaten_variable<phi::DenseTensor> {
  explicit topsaten_variable(const phi::DenseTensor& tensor) {
    value = CreateTopsatenTensor(tensor);
  }

  topsatenTensor value;
};

template <>
struct topsaten_variable<paddle::optional<phi::DenseTensor>> {
  explicit topsaten_variable(
      const paddle::optional<phi::DenseTensor>& opt_tensor) {
    value = OptionalTensorToTopsatenTensor(opt_tensor);
  }

  topsatenTensor value;
};

template <>
struct topsaten_variable<std::vector<phi::DenseTensor>> {
  explicit topsaten_variable(const std::vector<phi::DenseTensor>& tensor_list) {
    for (int64_t i = 0; i < tensor_list.size(); ++i) {
      value.emplace_back(CreateTopsatenTensor(tensor_list[i]));
    }
  }

  std::vector<topsatenTensor> value;
};

template <>
struct topsaten_variable<phi::Scalar> {
  explicit topsaten_variable(const phi::Scalar& scalar) {
    value = ScalarToTopsatenScalar(scalar);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<paddle::optional<phi::Scalar>> {
  explicit topsaten_variable(const paddle::optional<phi::Scalar>& opt_scalar) {
    value = OptionalScalarToTopsatenScalar(opt_scalar);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<phi::IntArray> {
  explicit topsaten_variable(const phi::IntArray& array) {
    value = IntArrayToTopsatenSize(array);
  }

  topsatenSize_t value;
};

template <>
struct topsaten_variable<paddle::optional<phi::IntArray>> {
  explicit topsaten_variable(const paddle::optional<phi::IntArray>& array) {
    value = OptionalIntArrayToTopsatenSize(array);
  }

  topsatenSize_t value;
};

template <>
struct topsaten_variable<std::vector<int64_t>> {
  explicit topsaten_variable(const std::vector<int64_t>& array) {
    value = IntArrayToTopsatenSize(array);
  }

  topsatenSize_t value;
};

template <>
struct topsaten_variable<phi::DataType> {
  explicit topsaten_variable(const phi::DataType& data_type) {
    value = DataTypeToTopsatenDataType(data_type);
  }

  topsatenDataType_t value;
};

template <>
struct topsaten_variable<std::pair<uint64_t, uint64_t>> {
  explicit topsaten_variable(const std::pair<uint64_t, uint64_t>& gen_data) {
    value.seed = gen_data.first;
    value.offset = gen_data.second;
  }

  topsatenGenerator_t value;
};

#define ATEN_OP_CALL_MAYBE_SYNC(func, ctx)                                    \
  do {                                                                        \
    VLOG(6) << "[AOT_KERNEL] Call tops aten op: "                             \
            << GetOpNameFromCallStatement(std::string(#func));                \
    GCU_AOT_KERNEL_TRACE(GetOpNameFromCallStatement(std::string(#func)));     \
    auto status = (func);                                                     \
    PADDLE_ENFORCE_EQ(                                                        \
        status,                                                               \
        TOPSATEN_STATUS_SUCCESS,                                              \
        phi::errors::Fatal("Failed to call aten op, get error: %d", status)); \
    GcuOpMaybeStreamSync(ctx);                                                \
  } while (false)

#define DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(namespace, topsatenop) \
  template <typename... Args>                                          \
  auto launch_##topsatenop(const phi::CustomContext& dev_ctx,          \
                           phi::DenseTensor& out,                      \
                           const Args&... args) {                      \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());         \
    auto xout = topsaten_variable(out);                                \
    topsatenStatus_t status;                                           \
    {                                                                  \
      GCU_AOT_KERNEL_TRACE(#topsatenop);                               \
      status = namespace ::topsatenop(                                 \
          xout.value, topsaten_variable(args).value..., stream);       \
      GcuOpMaybeStreamSync(dev_ctx);                                   \
    }                                                                  \
    return status;                                                     \
  }

#define DEFINE_LAUNCH_TOPSATENOP(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(topsaten, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_VLLM(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(topsvllm, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_OUT2_WITH_NAMESPACE(namespace, topsatenop)    \
  template <typename... Args>                                                  \
  auto launch_##topsatenop(const phi::CustomContext& dev_ctx,                  \
                           phi::DenseTensor& out1,                             \
                           phi::DenseTensor& out2,                             \
                           const Args&... args) {                              \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());                 \
    auto xout1 = topsaten_variable(out1);                                      \
    auto xout2 = topsaten_variable(out2);                                      \
    topsatenStatus_t status;                                                   \
    {                                                                          \
      GCU_AOT_KERNEL_TRACE(#topsatenop);                                       \
      status = namespace ::topsatenop(                                         \
          xout1.value, xout2.value, topsaten_variable(args).value..., stream); \
      GcuOpMaybeStreamSync(dev_ctx);                                           \
    }                                                                          \
    return status;                                                             \
  }

#define DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_OUT2_WITH_NAMESPACE(topsaten, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_OUT2_VLLM(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_OUT2_WITH_NAMESPACE(topsvllm, topsatenop)

#define LAUNCH_TOPSATENOP(topsatenop, dev_ctx, topsatenop_args...)         \
  {                                                                        \
    auto op_info = [&]() -> std::string {                                  \
      return custom_kernel::GetOpInfo(                                     \
          #topsatenop,                                                     \
          ##topsatenop_args,                                               \
          static_cast<topsStream_t>(dev_ctx.stream()));                    \
    };                                                                     \
    VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info(); \
    auto status =                                                          \
        custom_kernel::launch_##topsatenop(dev_ctx, ##topsatenop_args);    \
    PADDLE_ENFORCE_EQ(                                                     \
        status,                                                            \
        TOPSATEN_STATUS_SUCCESS,                                           \
        phi::errors::Fatal(                                                \
            "Failed to call aten op, get error: %d, details: %s",          \
            status,                                                        \
            op_info().c_str()));                                           \
    VLOG(6) << "Launch tops aten op sucessfully, details:" << op_info();   \
  }

#define LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenop, dev_ctx, ...)          \
  {                                                                            \
    VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << #topsatenop;   \
    auto status = topsaten::topsatenop(                                        \
        __VA_ARGS__, static_cast<topsStream_t>(dev_ctx.stream()));             \
    PADDLE_ENFORCE_EQ(                                                         \
        status,                                                                \
        TOPSATEN_STATUS_SUCCESS,                                               \
        phi::errors::Fatal("Failed to call aten op, get error: %d.", status)); \
    VLOG(6) << "Launch tops aten op sucessfully, details:" << #topsatenop;     \
    GcuOpMaybeStreamSync(dev_ctx);                                             \
  }
}  // namespace

// binary op
DEFINE_LAUNCH_TOPSATENOP(topsatenAdd)
DEFINE_LAUNCH_TOPSATENOP(topsatenSub)
DEFINE_LAUNCH_TOPSATENOP(topsatenMul)
DEFINE_LAUNCH_TOPSATENOP(topsatenDiv)
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseAnd)
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseOr)
DEFINE_LAUNCH_TOPSATENOP(topsatenGt)
DEFINE_LAUNCH_TOPSATENOP(topsatenGe)
DEFINE_LAUNCH_TOPSATENOP(topsatenNe)
DEFINE_LAUNCH_TOPSATENOP(topsatenEq)
DEFINE_LAUNCH_TOPSATENOP(topsatenLe)
DEFINE_LAUNCH_TOPSATENOP(topsatenLt)
DEFINE_LAUNCH_TOPSATENOP(topsatenMatmul)

// unary op
DEFINE_LAUNCH_TOPSATENOP(topsatenCos)
DEFINE_LAUNCH_TOPSATENOP(topsatenRsqrt)
DEFINE_LAUNCH_TOPSATENOP(topsatenSin)
DEFINE_LAUNCH_TOPSATENOP(topsatenLog)
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseNot)
DEFINE_LAUNCH_TOPSATENOP(topsatenPow)

// activation op
DEFINE_LAUNCH_TOPSATENOP(topsatenRelu)
DEFINE_LAUNCH_TOPSATENOP(topsatenSilu)
// DEFINE_LAUNCH_TOPSATENOP(topsatenSwish) same with Silu
DEFINE_LAUNCH_TOPSATENOP(topsatenSigmoid)
DEFINE_LAUNCH_TOPSATENOP(topsatenHardswish)
DEFINE_LAUNCH_TOPSATENOP(topsatenHardsigmoid)

// reduce op
DEFINE_LAUNCH_TOPSATENOP(topsatenAny)
DEFINE_LAUNCH_TOPSATENOP(topsatenMean)
DEFINE_LAUNCH_TOPSATENOP(topsatenSum)
DEFINE_LAUNCH_TOPSATENOP(topsatenMax)
DEFINE_LAUNCH_TOPSATENOP(topsatenMin)
DEFINE_LAUNCH_TOPSATENOP(topsatenProd)
DEFINE_LAUNCH_TOPSATENOP(topsatenCumsum)

// DEFINE_LAUNCH_TOPSATENOP(topsatenExpand)
DEFINE_LAUNCH_TOPSATENOP(topsatenFull)
DEFINE_LAUNCH_TOPSATENOP(topsatenTril)
DEFINE_LAUNCH_TOPSATENOP(topsatenTriu)
DEFINE_LAUNCH_TOPSATENOP(topsatenWhere)
DEFINE_LAUNCH_TOPSATENOP(topsatenEmbedding)
DEFINE_LAUNCH_TOPSATENOP(topsatenSoftmaxForward)
// DEFINE_LAUNCH_TOPSATENOP(topsatenTo)
// DEFINE_LAUNCH_TOPSATENOP(topsatenStack)

DEFINE_LAUNCH_TOPSATENOP(topsatenArgSort)
DEFINE_LAUNCH_TOPSATENOP(topsatenArgmax)
DEFINE_LAUNCH_TOPSATENOP(topsatenIndex)
DEFINE_LAUNCH_TOPSATENOP(topsatenIndexSelect)
DEFINE_LAUNCH_TOPSATENOP(topsatenScatter)
DEFINE_LAUNCH_TOPSATENOP(topsatenGather)
DEFINE_LAUNCH_TOPSATENOP(topsatenIndexPut)

DEFINE_LAUNCH_TOPSATENOP(topsatenOnes)
DEFINE_LAUNCH_TOPSATENOP(topsatenZeros)
DEFINE_LAUNCH_TOPSATENOP(topsatenArange)
DEFINE_LAUNCH_TOPSATENOP(topsatenTile)
DEFINE_LAUNCH_TOPSATENOP(topsatenUpsampleNearest2d)

// distributions
DEFINE_LAUNCH_TOPSATENOP(topsatenRngUniform)
DEFINE_LAUNCH_TOPSATENOP(topsatenNormal)
DEFINE_LAUNCH_TOPSATENOP(topsatenMultinomial)

DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenTopk)
DEFINE_LAUNCH_TOPSATENOP(topsatenDropout)
DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenNativeDropout)

// vllm
DEFINE_LAUNCH_TOPSATENOP_OUT2_VLLM(topsvllmRotaryEmbedding)
DEFINE_LAUNCH_TOPSATENOP_OUT2_VLLM(topsvllmFusedAddRmsNorm)
DEFINE_LAUNCH_TOPSATENOP_VLLM(topsvllmMemoryEfficientAttention)
DEFINE_LAUNCH_TOPSATENOP_VLLM(topsvllmRmsNorm)
DEFINE_LAUNCH_TOPSATENOP_VLLM(topsvllmSiluAndMul)

}  // namespace custom_kernel
