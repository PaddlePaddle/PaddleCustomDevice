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

#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kernels/funcs/sdaa_funcs.h"
#include "kernels/profiler/sdaa_wrapper.h"
#include "paddle/phi/extension.h"
#include "tecodnn_custom.h"  // NOLINT

#define kSliceMaxNum 8
#define StoragePropertiesCHWN 0
using SDAAStorageProperties = phi::NPUStorageProperties;
namespace custom_kernel {

using Context = phi::CustomContext;
using DataType = phi::DataType;
using DataLayout = phi::DataLayout;

template <typename T>
class MPTypeTrait {
 public:
  using Type = T;
};

template <>
class MPTypeTrait<phi::dtype::float16> {
 public:
  using Type = float;
};

template <>
class MPTypeTrait<phi::dtype::bfloat16> {
 public:
  using Type = float;
};

enum class Storage_Format {
  CHWN = 0,
};
enum class Convert_TF {
  NCHW2NHWC = 0,
  NCHW2CHWN = 1,
  NHWC2NCHW = 2,
  NHWC2CHWN = 3,
  CHWN2NCHW = 4,
  CHWN2NHWC = 5,
  NHWC2NWHC = 6,
};

enum class ActivationMode {
  sigmoid = 0,
  relu = 1,
  tanh = 2,
  elu = 3,
  clipped_relu = 4,
  gelu = 5,
  leaky_relu = 6,
  selu = 7,
  relu6 = 8,
  silu = 9,
  gelu_approximate = 10,
};

enum class NanPropagation {
  not_propagate_nan = 0,
  propagate_nan = 1,
};

enum class UnaryOpMode {
  // without alpha
  LOG = 0,
  EXP = 1,
  SQRT = 2,
  RSQRT = 3,
  SQUARE = 4,
  SIN = 5,
  COS = 6,
  TANH = 7,
  CEIL = 8,
  FLOOR = 9,
  FABS = 10,
  // with alpha
  ADD_A = 11,
  SUB_A = 12,
  MUL_A = 13,
  DIV_A = 14,
  RDIV = 15,
  POW = 16
};

enum class CompareType {
  Equal = 1,
  GreaterThan,
  LessThan,
  GreaterEqual,
  LessEqual,
  NotEqual
};

enum class OpTensorMode {
  Add = 0,
  Mul = 1,
  Min = 2,
  Max = 3,
  Sqrt = 4,
  Not = 5
};

enum class TensorFormat {
  NCHW = 0,
  NHWC = 1,
  CHWN = 2,
  NWHC = 3,
  Undefined = 4
};

enum class TensorLogicType { all = 1, any = 2 };

enum class BitwiseOpType { And = 1, Or = 2, Xor = 3, Not = 4 };

enum class LogicalOpType { And = 1, Or = 2, Xor = 3, Not = 4 };

namespace sdaa_ops {
inline DataTypes_t ToExtendDataType(const DataType& dtype) {
  DataTypes_t dt = DATA_FLOAT;
  switch (dtype) {
    case DataType::FLOAT16:
      dt = DATA_HALF;
      break;
    case DataType::FLOAT32:
      dt = DATA_FLOAT;
      break;
    case DataType::INT8:
      dt = DATA_INT8;
      break;
    case DataType::INT16:
      dt = DATA_INT16;
      break;
    case DataType::INT32:
      dt = DATA_INT32;
      break;
    default:
      break;
  }
  return dt;
}

inline tecodnnDataType_t ToTecodnnDataType(const DataType& dtype) {
  tecodnnDataType_t dt = TECODNN_DATA_FLOAT;
  switch (dtype) {
    case DataType::FLOAT16:
      dt = TECODNN_DATA_HALF;
      break;
    case DataType::FLOAT32:
      dt = TECODNN_DATA_FLOAT;
      break;
    case DataType::FLOAT64:
      dt = TECODNN_DATA_DOUBLE;
      break;
    case DataType::INT8:
      dt = TECODNN_DATA_INT8;
      break;
    case DataType::INT16:
      dt = TECODNN_DATA_INT16;
      break;
    case DataType::INT32:
      dt = TECODNN_DATA_INT32;
      break;
    case DataType::INT64:
      dt = TECODNN_DATA_INT64;
      break;
    case DataType::BOOL:
      dt = TECODNN_DATA_UINT8;
      break;
    case DataType::UINT8:
      dt = TECODNN_DATA_UINT8;
      break;
    default:
      break;
  }
  return dt;
}

template <typename T>
class MPTypeTrait {
 public:
  using Type = T;
};

template <>
class MPTypeTrait<phi::dtype::float16> {
 public:
  using Type = float;
};

// Tensor dtype to TCCL dtype
inline tcclDataType_t ToTcclDataType(const DataType& dtype) {
  if (dtype == DataType::FLOAT32) {
    return tcclFloat;
  } else if (dtype == DataType::FLOAT16) {
    return tcclHalf;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Datatype %s in tccl is not supported.", dtype));
  }
}

inline void ExtractNCWHD(const phi::DDim& dims,
                         const DataLayout& data_layout,
                         int* N,
                         int* C,
                         int* H,
                         int* W,
                         int* D) {
  *N = dims[0];
  if (dims.size() == 2) {
    *C = dims[1];
    *H = 1;
    *W = 1;
    *D = 1;
  } else {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
    *H = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *W = dims.size() > 3
             ? (data_layout == DataLayout::kNCHW ? dims[3] : dims[2])
             : 1;
    *D = dims.size() > 4
             ? (data_layout == DataLayout::kNCHW ? dims[4] : dims[3])
             : 1;
  }
}

inline tecodnnTensorFormat_t ToTecodnnTensorFormat(
    const DataLayout& tensor_format) {
  tecodnnTensorFormat_t tf = TECODNN_TENSOR_NHWC;
  switch (tensor_format) {
    case DataLayout::NCHW:
      tf = TECODNN_TENSOR_NCHW;
      break;
    case DataLayout::NHWC:
      tf = TECODNN_TENSOR_NHWC;
      break;
    default:
      break;
  }
  return tf;
}

inline tecodnnTensorFormat_t GetTecodnnTF(const TensorFormat TF) {
  tecodnnTensorFormat_t tf = TECODNN_TENSOR_NHWC;
  switch (TF) {
    case TensorFormat::NCHW:
      tf = TECODNN_TENSOR_NCHW;
      break;
    case TensorFormat::NHWC:
      tf = TECODNN_TENSOR_NHWC;
      break;
    case TensorFormat::CHWN:
      tf = TECODNN_TENSOR_CHWN;
      break;
    case TensorFormat::NWHC:
      tf = TECODNN_TENSOR_NWHC;
      break;
    default:
      break;
  }
  return tf;
}

const std::map<Convert_TF, std::pair<TensorFormat, TensorFormat>>
    tecodnnTensorFormatMap = {
        {Convert_TF::NCHW2NHWC, {TensorFormat::NCHW, TensorFormat::NHWC}},
        {Convert_TF::NCHW2CHWN, {TensorFormat::NCHW, TensorFormat::CHWN}},
        {Convert_TF::NHWC2NCHW, {TensorFormat::NHWC, TensorFormat::NCHW}},
        {Convert_TF::NHWC2CHWN, {TensorFormat::NHWC, TensorFormat::CHWN}},
        {Convert_TF::CHWN2NCHW, {TensorFormat::CHWN, TensorFormat::NCHW}},
        {Convert_TF::CHWN2NHWC, {TensorFormat::CHWN, TensorFormat::NHWC}},
        {Convert_TF::NHWC2NWHC, {TensorFormat::NHWC, TensorFormat::NWHC}},
};

const std::map<ActivationMode, tecodnnActivationMode_t>
    tecodnnActivationModeMap = {
        {ActivationMode::sigmoid, TECODNN_ACTIVATION_SIGMOID},
        {ActivationMode::relu, TECODNN_ACTIVATION_RELU},
        {ActivationMode::tanh, TECODNN_ACTIVATION_TANH},
        {ActivationMode::elu, TECODNN_ACTIVATION_ELU},
        {ActivationMode::clipped_relu, TECODNN_ACTIVATION_CLIPPED_RELU},
        {ActivationMode::gelu, TECODNN_ACTIVATION_GELU},
        {ActivationMode::leaky_relu, TECODNN_ACTIVATION_LEAKYRELU},
        {ActivationMode::selu, TECODNN_ACTIVATION_SELU},
        {ActivationMode::relu6, TECODNN_ACTIVATION_RELU6},
        {ActivationMode::silu, TECODNN_ACTIVATION_SILU},
        {ActivationMode::gelu_approximate, TECODNN_ACTIVATION_GELU_APPROXIMATE},
};

const std::map<NanPropagation, tecodnnNanPropagation_t>
    tecodnnNanPropagationMap = {
        {NanPropagation::not_propagate_nan, TECODNN_NOT_PROPAGATE_NAN},
        {NanPropagation::propagate_nan, TECODNN_PROPAGATE_NAN},
};

const std::map<UnaryOpMode, tecodnnUnaryOpsMode_t> tecodnnUnaryOpsModeMap = {
    {UnaryOpMode::LOG, TECODNN_BATCH_LOG},
    {UnaryOpMode::EXP, TECODNN_BATCH_EXP},
    {UnaryOpMode::SQRT, TECODNN_BATCH_SQRT},
    {UnaryOpMode::RSQRT, TECODNN_BATCH_RSQRT},
    {UnaryOpMode::SQUARE, TECODNN_BATCH_SQUARE},
    {UnaryOpMode::SIN, TECODNN_BATCH_SIN},
    {UnaryOpMode::COS, TECODNN_BATCH_COS},
    {UnaryOpMode::TANH, TECODNN_BATCH_TANH},
    {UnaryOpMode::CEIL, TECODNN_BATCH_CEIL},
    {UnaryOpMode::FLOOR, TECODNN_BATCH_FLOOR},
    {UnaryOpMode::FABS, TECODNN_BATCH_FABS},
    {UnaryOpMode::ADD_A, TECODNN_BATCH_ADD_A},
    {UnaryOpMode::SUB_A, TECODNN_BATCH_SUB_A},
    {UnaryOpMode::MUL_A, TECODNN_BATCH_MUL_A},
    {UnaryOpMode::DIV_A, TECODNN_BATCH_DIV_A},
    {UnaryOpMode::RDIV, TECODNN_BATCH_RDIV},
    {UnaryOpMode::POW, TECODNN_BATCH_POW},
};

const std::map<OpTensorMode, tecodnnOpTensorOp_t> tecodnnOpTensorModeMap = {
    {OpTensorMode::Add, TECODNN_OP_TENSOR_ADD},
    {OpTensorMode::Mul, TECODNN_OP_TENSOR_MUL},
    {OpTensorMode::Min, TECODNN_OP_TENSOR_MIN},
    {OpTensorMode::Max, TECODNN_OP_TENSOR_MAX},
    {OpTensorMode::Sqrt, TECODNN_OP_TENSOR_SQRT},
    {OpTensorMode::Not, TECODNN_OP_TENSOR_NOT},
};

const std::map<std::vector<int>, Convert_TF> TransposeModeMap = {
    {{1, 0}, Convert_TF::NHWC2NWHC},
    {{0, 2, 1}, Convert_TF::NHWC2NWHC},
    {{1, 0, 2}, Convert_TF::NHWC2NWHC},
    {{2, 0, 1}, Convert_TF::NHWC2NCHW},
    {{1, 2, 0}, Convert_TF::NCHW2NHWC},

    {{0, 2, 3, 1}, Convert_TF::NCHW2NHWC},
    {{1, 2, 3, 0}, Convert_TF::NCHW2CHWN},
    {{0, 3, 1, 2}, Convert_TF::NHWC2NCHW},
    {{3, 1, 2, 0}, Convert_TF::NHWC2CHWN},
    {{0, 2, 1, 3}, Convert_TF::NHWC2NWHC},
    {{3, 0, 1, 2}, Convert_TF::CHWN2NCHW},

    {{0, 1, 3, 4, 2}, Convert_TF::NCHW2NHWC},
    {{2, 3, 4, 0, 1}, Convert_TF::NCHW2CHWN},
    {{0, 1, 4, 2, 3}, Convert_TF::NHWC2NCHW},
    {{4, 2, 3, 0, 1}, Convert_TF::NHWC2CHWN},
    {{0, 1, 3, 2, 4}, Convert_TF::NHWC2NWHC},

    {{0, 3, 4, 1, 2}, Convert_TF::NCHW2NHWC},
    {{1, 2, 3, 4, 0}, Convert_TF::NCHW2CHWN},
    {{0, 4, 1, 2, 3}, Convert_TF::NHWC2NCHW},
    {{4, 1, 2, 3, 0}, Convert_TF::NHWC2CHWN},
    {{0, 3, 1, 2, 4}, Convert_TF::NHWC2NWHC},
    {{4, 0, 1, 2, 3}, Convert_TF::CHWN2NCHW},

    {{0, 2, 3, 4, 1}, Convert_TF::NCHW2NHWC},
    {{0, 2, 3, 1, 4}, Convert_TF::NHWC2NWHC},

    {{3, 4, 1, 2, 0}, Convert_TF::NHWC2CHWN},
    {{0, 2, 1, 3, 4}, Convert_TF::NHWC2NWHC},
    {{3, 4, 0, 1, 2}, Convert_TF::CHWN2NCHW}};

void doMeanTensor(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& reduce_dims,
                  phi::DenseTensor* y);

void doSumTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& reduce_dims,
                 phi::DenseTensor* y);

void doProdTensor(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& reduce_dims,
                  phi::DenseTensor* y);

void doMinTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& reduce_dims,
                 phi::DenseTensor* y);

void doMaxTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& reduce_dims,
                 phi::DenseTensor* y);

void doTransformTensor(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       Convert_TF convert_tf,
                       phi::DenseTensor* y);

void doCastTensor(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* y);

void doAddTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 float alpha,
                 float beta,
                 phi::DenseTensor* out);

void doActivationForward(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         double factor,
                         ActivationMode activation_mode,
                         NanPropagation nan_propagate,
                         phi::DenseTensor* out);

void doActivationBackward(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          double factor,
                          ActivationMode activation_mode,
                          NanPropagation nan_propagate,
                          phi::DenseTensor* dx);

void doUnaryOpTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     UnaryOpMode unaryOpMode,
                     phi::DenseTensor* out);

void doScaleTensor(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   float scale,
                   float bias,
                   bool inplace,
                   bool bias_flag,
                   phi::DenseTensor* out);

void doNegTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out);

void doCompareTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     CompareType tct,
                     phi::DenseTensor* out);

void doOpTensor(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& y,
                OpTensorMode opTensorMode,
                phi::DenseTensor* out);

void doElementAdd(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out);

void doElementSub(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out);

void doElementMul(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out);

void doElementDiv(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out);

tecodnnTensorDescriptor_t GetTecodnnTensorDesc(
    const std::vector<int>& dims,
    const DataType& dtype,
    TensorFormat tf = TensorFormat::Undefined);

tecodnnTensorDescriptor_t GetTecodnnBoolTensorDesc(const std::vector<int>& dims,
                                                   TensorFormat tf);

void doReciprocalTensor(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        phi::DenseTensor* out);

void doSoftmaxForward(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      int axis,
                      bool high_precision,
                      phi::DenseTensor* out);

void doSoftmaxBackward(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       int axis,
                       bool high_precision,
                       phi::DenseTensor* dx);

void doLogSoftmaxForward(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         int axis,
                         phi::DenseTensor* out);

void doLogSoftmaxBackward(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          int axis,
                          phi::DenseTensor* dx);

phi::DDim doDimPermute(const phi::DenseTensor& x, Convert_TF convert_tf);

template <typename T>
void doSliceTensor(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const std::vector<T>& axes,
                   const std::vector<T>& starts,
                   const std::vector<T>& ends,
                   const std::vector<T>& strides,
                   const std::vector<int64_t>& decrease_axis,
                   phi::DenseTensor* out);

void doPaddingTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<std::vector<int>>& Paddings,
                     phi::DenseTensor* out);

void doTransposeTensor(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const std::vector<int>& axis,
                       phi::DenseTensor* out);

void doLogicTensor(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const std::vector<int64_t>& axis_reduce,
                   TensorLogicType TLT,
                   phi::DenseTensor* out);

void doConcatTensor(const Context& dev_ctx,
                    const std::vector<const phi::DenseTensor*>& x,
                    int axis,
                    phi::DenseTensor* out);

void doScatterTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& index,
                     const phi::DenseTensor& updates,
                     bool overwrite,
                     phi::DenseTensor* out);

void doSplitTensor(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   std::vector<phi::DenseTensor*> outs);

void doExpandTensor(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out);

void doNearestInterpolateForward(const Context& dev_ctx,
                                 const phi::DenseTensor& x,
                                 const float ratio_w,
                                 const float ratio_h,
                                 const float ratio_d,
                                 const bool align_corners,
                                 phi::DenseTensor* out);

void doNearestInterpolateBackward(const Context& dev_ctx,
                                  const phi::DenseTensor& out,
                                  const float ratio_w,
                                  const float ratio_h,
                                  const float ratio_d,
                                  const bool align_corners,
                                  phi::DenseTensor* dx);

void doBitwiseBinaryOpTensor(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             BitwiseOpType bitwiseType,
                             phi::DenseTensor* out);

void doBitwiseUnaryOpTensor(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            BitwiseOpType bitwiseType,
                            phi::DenseTensor* out);
int64_t doAddStorageProperties(
    const Context& dev_ctx,
    phi::DenseTensor* tensor,
    SDAAStorageProperties& storage_properties);  // NOLINT

void doIsnanOp(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out);

void doLogicalOpTensor(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       LogicalOpType logicaltype,
                       phi::DenseTensor* out);

void doLogicalNotOpTensor(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          phi::DenseTensor* out);

void swapTensorData(const Context& dev_ctx,
                    const phi::DenseTensor& in,
                    SDAAStorageProperties& storage_properties);  // NOLINT

void doAtanTensor(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out);

template <typename T>
void doFillTensor(const Context& dev_ctx,
                  T val,
                  phi::DataType dtype,
                  phi::DenseTensor* out) {
  auto handle = custom_kernel::GetHandleFromCTX(dev_ctx);
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  tecodnnTensorDescriptor_t Desc;
  if (out->dtype() == phi::DataType::BOOL) {
    Desc = custom_kernel::sdaa_ops::GetTecodnnBoolTensorDesc(
        out_dims, TensorFormat::Undefined);
  } else {
    Desc = custom_kernel::sdaa_ops::GetTecodnnTensorDesc(
        out_dims, out->dtype(), TensorFormat::Undefined);
  }
  TECODNN_CHECK(tecodnnSetTensor(handle, Desc, out->data(), &val));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

std::vector<int64_t> GetReduceDimAxis(const phi::DDim& in,
                                      const phi::DDim& out,
                                      int axis);

void BatchNormFunc(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& mean,
                   const phi::DenseTensor& variance,
                   const phi::DenseTensor& scale,
                   const phi::DenseTensor& bias,
                   float momentum,
                   float epsilon,
                   bool training,
                   const std::string& data_layout_str,
                   phi::DenseTensor* y,
                   phi::DenseTensor* mean_out,
                   phi::DenseTensor* variance_out,
                   phi::DenseTensor* saved_mean,
                   phi::DenseTensor* saved_variance);

void doMemsetTensor(const Context& dev_ctx,
                    const int value,
                    phi::DenseTensor* tensor);

void GetReduceDimReduceAll(const std::vector<int>& axis_dims,
                           int input_dims_size,
                           bool reduce_all,
                           std::vector<int>* reduce_dims);

template <typename T>
void doScatterNdAdd(const Context& ctx,
                    const phi::DenseTensor& index,
                    const phi::DenseTensor& updates,
                    phi::DenseTensor* out);

}  // namespace sdaa_ops
}  // namespace custom_kernel
