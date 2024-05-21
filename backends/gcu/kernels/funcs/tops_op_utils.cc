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

#include "kernels/funcs/tops_op_utils.h"

#include <unordered_map>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace custom_kernel {

topsopTensor::topsopTensor(topsopSize_t* dims,
                           topsopSize_t* strides,
                           topsopDataType_t dtype,
                           topsopDeviceMemHandle_t memhandle) {
  CHECK_TOPS_OP_CALL(topsopCreateTensor(
      dims, strides, dtype, memhandle, &tops_op_tensor_handle_));
  PADDLE_ENFORCE_NOT_NULL(tops_op_tensor_handle_);
}

topsopDeviceMemHandle_t topsopTensor::GetTensorData() const {
  topsopDeviceMemHandle_t memhandle;
  CHECK_TOPS_OP_CALL(topsopGetTensorData(tops_op_tensor_handle_, &memhandle));
  return memhandle;
}

topsopSize_t topsopTensor::GetTensorShape() const {
  topsopSize_t dims;
  CHECK_TOPS_OP_CALL(topsopGetTensorShape(tops_op_tensor_handle_, &dims));
  return dims;
}

topsopSize_t topsopTensor::GetTensorStrides() const {
  topsopSize_t strides;
  CHECK_TOPS_OP_CALL(topsopGetTensorStride(tops_op_tensor_handle_, &strides));
  return strides;
}

topsopDataType_t topsopTensor::GetTensorDataType() const {
  topsopDataType_t dtype;
  CHECK_TOPS_OP_CALL(topsopGetTensorDataType(tops_op_tensor_handle_, &dtype));
  return dtype;
}

// This function gets total elements number of the tensor.
int64_t topsopTensor::GetTensorElementNums() const {
  int64_t nums = 0;
  CHECK_TOPS_OP_CALL(topsopGetTensorElementNums(tops_op_tensor_handle_, &nums));
  return nums;
}

// This function gets bytes per element of the tensor.
int64_t topsopTensor::GetTensorBPE() const {
  int64_t itemsize = 1;
  CHECK_TOPS_OP_CALL(topsopGetTensorBPE(tops_op_tensor_handle_, &itemsize));
  return itemsize;
}

std::string topsopTensor::ToString() const {
  std::stringstream ss;
  ss << "topsopTensor<" << tops_op_tensor_handle_ << ">";
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const topsopTensor& tensor) {
  os << tensor.ToString();
  return os;
}

std::ostream& operator<<(std::ostream& os, const topsopTensorHandle_t& tensor) {
  std::stringstream ss;
  ss << "topsopTensorHandle<";
  if (tensor != nullptr) {
    topsopSize_t dims;
    CHECK_TOPS_OP_CALL(topsopGetTensorShape(tensor, &dims));
    topsopDataType_t dtype;
    CHECK_TOPS_OP_CALL(topsopGetTensorDataType(tensor, &dtype));
    ss << static_cast<const void*>(tensor) << ", " << dtype << ", ";
    ss << "Shape(" << dims << ")";
  } else {
    ss << "NOT_INITED";
  }
  ss << ">";
  os << ss.str();
  return os;
}

std::ostream& operator<<(std::ostream& os, const topsopDataType_t& dtype) {
  static const std::unordered_map<topsopDataType_t, std::string>
      kTopsopDtypeToStr = {{TOPSOP_DATA_I8, "TOPSOP_DATA_I8"},
                           {TOPSOP_DATA_U8, "TOPSOP_DATA_U8"},
                           {TOPSOP_DATA_I16, "TOPSOP_DATA_I16"},
                           {TOPSOP_DATA_U16, "TOPSOP_DATA_U16"},
                           {TOPSOP_DATA_FP16, "TOPSOP_DATA_FP16"},
                           {TOPSOP_DATA_BF16, "TOPSOP_DATA_BF16"},
                           {TOPSOP_DATA_I32, "TOPSOP_DATA_I32"},
                           {TOPSOP_DATA_U32, "TOPSOP_DATA_U32"},
                           {TOPSOP_DATA_FP32, "TOPSOP_DATA_FP32"},
                           {TOPSOP_DATA_EF32, "TOPSOP_DATA_EF32"},
                           {TOPSOP_DATA_TF32, "TOPSOP_DATA_TF32"},
                           {TOPSOP_DATA_I64, "TOPSOP_DATA_I64"},
                           {TOPSOP_DATA_U64, "TOPSOP_DATA_U64"},
                           {TOPSOP_DATA_F64, "TOPSOP_DATA_F64"},
                           {TOPSOP_DATA_PRED, "TOPSOP_DATA_PRED"},
                           {TOPSOP_DATA_I4, "TOPSOP_DATA_I4"}};
  os << "topsopDataType_t(";
  if (kTopsopDtypeToStr.count(dtype) > 0) {
    os << kTopsopDtypeToStr.at(dtype);
  } else {
    os << "UnknownType:" << static_cast<int64_t>(dtype);
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const topsopStatus_t& status) {
  static const std::unordered_map<topsopStatus_t, std::string>
      kTopsopStatusToStr = {
          {TOPSOP_STATUS_SUCCESS, "TOPSOP_STATUS_SUCCESS"},
          {TOPSOP_STATUS_ALLOC_FAILED, "TOPSOP_STATUS_ALLOC_FAILED"},
          {TOPSOP_STATUS_BAD_PARAM, "TOPSOP_STATUS_BAD_PARAM"},
          {TOPSOP_STATUS_NOT_SUPPORT, "TOPSOP_STATUS_NOT_SUPPORT"},
          {TOPSOP_STATUS_INTERNAL_ERROR, "TOPSOP_STATUS_INTERNAL_ERROR"},
          {TOPSOP_STATUS_RUNTIME_ERROR, "TOPSOP_STATUS_RUNTIME_ERROR"},
          {TOPSOP_STATUS_EXECUTE_ERROR, "TOPSOP_STATUS_EXECUTE_ERROR"}};
  os << "topsopStatus_t(";
  if (kTopsopStatusToStr.count(status) > 0) {
    os << kTopsopStatusToStr.at(status) << ":" << static_cast<int64_t>(status);
  } else {
    os << "UnknownStatus:" << static_cast<int64_t>(status);
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const topsopSize_t& size) {
  auto dims = std::vector<int64_t>(size.data, size.data + size.len);
  os << "topsopSize_t(" << VectorToStr<int64_t>(dims) << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const topsopScalar_t& scalar) {
  os << TopsopScalarToString(scalar);
  return os;
}

topsopTensor CreateTopsopTensor(const phi::DenseTensor& tensor) {
  return topsopTensor(CreateTopsopTensorHandle(tensor));
}

topsopTensorHandle_t CreateTopsopTensorHandle(const phi::DenseTensor& tensor) {
  PADDLE_ENFORCE_EQ(
      tensor.initialized(),
      true,
      phi::errors::InvalidArgument("Gcu tensor should be inited"));
  PADDLE_ENFORCE_EQ(
      tensor.place().GetDeviceType(),
      "gcu",
      phi::errors::InvalidArgument("Not gcu tensor, current tensor place: %s",
                                   tensor.place().GetDeviceType().c_str()));

  std::vector<int64_t> tensor_dims = phi::vectorize(tensor.dims());
  std::vector<int64_t> strides = phi::vectorize(tensor.strides());
  int64_t rank = tensor_dims.size();
  if (rank == 0) {
    rank = 1;
    tensor_dims = {1};
    strides = {1};
  }
  auto xdims = topsopSize_t{tensor_dims.data(), rank};
  auto xstrides = topsopSize_t{strides.data(), rank};
  auto xdtype = DataTypeToTopsopDataType(tensor.dtype());
  VLOG(6) << "topsopCreateTensor, input tensor:" << TensorToString(tensor);

  topsopTensorHandle_t tops_op_tensor_handle;
  CHECK_TOPS_OP_CALL(topsopCreateTensor(
      &xdims, &xstrides, xdtype, GcuDataPtr(tensor), &tops_op_tensor_handle));
  PADDLE_ENFORCE_NOT_NULL(tops_op_tensor_handle);
  VLOG(6) << "Create tensor handle successfully:" << tops_op_tensor_handle;
  return tops_op_tensor_handle;
}

topsopTensorHandle_t OptionalTensorToTopsopTensorHandle(
    const paddle::optional<phi::DenseTensor>& opt_tensor) {
  if (opt_tensor) {
    return CreateTopsopTensorHandle(opt_tensor.get());
  } else {
    return nullptr;
  }
}

topsopDataType_t DataTypeToTopsopDataType(const phi::DataType& dtype) {
  switch (dtype) {
    case phi::DataType::BOOL:
      return TOPSOP_DATA_PRED;
    case phi::DataType::UINT8:
      return TOPSOP_DATA_U8;
    case phi::DataType::INT8:
      return TOPSOP_DATA_I8;
    case phi::DataType::INT16:
      return TOPSOP_DATA_I16;
    case phi::DataType::INT32:
      return TOPSOP_DATA_I32;
    case phi::DataType::INT64:
      return TOPSOP_DATA_I64;
    case phi::DataType::FLOAT16:
      return TOPSOP_DATA_FP16;
    case phi::DataType::BFLOAT16:
      return TOPSOP_DATA_BF16;
    case phi::DataType::FLOAT32:
      return TOPSOP_DATA_FP32;
    case phi::DataType::FLOAT64:
      return TOPSOP_DATA_F64;
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported data type %s", phi::DataTypeToString(dtype).c_str()));
      return TOPSOP_DATA_FP32;
    }
  }
}

topsopScalar_t ScalarToTopsopScalar(const phi::Scalar& scalar_value) {
  topsopScalar_t xvalue;
  auto scalar_type = scalar_value.dtype();
  switch (scalar_type) {
    case phi::DataType::BOOL:
      xvalue.dtype = TOPSOP_DATA_PRED;
      xvalue.ival = scalar_value.to<bool>();
      break;
    case phi::DataType::UINT8:
      xvalue.dtype = TOPSOP_DATA_U8;
      xvalue.ival = scalar_value.to<uint8_t>();
      break;
    case phi::DataType::INT8:
      xvalue.dtype = TOPSOP_DATA_I8;
      xvalue.ival = scalar_value.to<int8_t>();
      break;
    case phi::DataType::INT16:
      xvalue.dtype = TOPSOP_DATA_I16;
      xvalue.ival = scalar_value.to<int16_t>();
      break;
    case phi::DataType::INT32:
      xvalue.dtype = TOPSOP_DATA_I32;
      xvalue.ival = scalar_value.to<int>();
      break;
    case phi::DataType::INT64:
      xvalue.dtype = TOPSOP_DATA_I64;
      xvalue.ival = scalar_value.to<int64_t>();
      break;
    case phi::DataType::FLOAT16:
      xvalue.dtype = TOPSOP_DATA_FP16;
      xvalue.fval = scalar_value.to<phi::float16>();
      break;
    case phi::DataType::BFLOAT16:
      xvalue.dtype = TOPSOP_DATA_BF16;
      xvalue.fval = scalar_value.to<phi::bfloat16>();
      break;
    case phi::DataType::FLOAT32:
      xvalue.dtype = TOPSOP_DATA_FP32;
      xvalue.fval = scalar_value.to<float>();
      break;
    case phi::DataType::FLOAT64:
      xvalue.dtype = TOPSOP_DATA_F64;
      xvalue.fval = scalar_value.to<double>();
      break;
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "ScalarToTopsopScalar, unsupported data type %s",
          phi::DataTypeToString(scalar_type).c_str()));
      break;
    }
  }
  return xvalue;
}

topsopScalar_t OptionalScalarToTopsopScalar(
    const paddle::optional<phi::Scalar>& opt_scalar) {
  if (opt_scalar) {
    return ScalarToTopsopScalar(opt_scalar.get());
  } else {
    return {};
  }
}

topsopSize_t IntArrayToTopsopSize(const phi::IntArray& int_array) {
  auto int_datas = int_array.GetData();
  return {int_datas.data(), static_cast<int64_t>(int_datas.size())};
}

topsopSize_t IntArrayToTopsopSize(const std::vector<int64_t>& int_array) {
  return {int_array.data(), static_cast<int64_t>(int_array.size())};
}

topsopSize_t OptionalIntArrayToTopsopSize(
    const paddle::optional<phi::IntArray>& opt_array) {
  if (opt_array) {
    return IntArrayToTopsopSize(opt_array.get());
  } else {
    return {};
  }
}

std::string TopsopScalarToString(const topsopScalar_t& scalar_value) {
  std::stringstream ss;
  ss << "topsopScalar<" << scalar_value.dtype << ", value(";
  switch (scalar_value.dtype) {
    case TOPSOP_DATA_I8:
    case TOPSOP_DATA_U8:
    case TOPSOP_DATA_I16:
    case TOPSOP_DATA_U16:
    case TOPSOP_DATA_I32:
    case TOPSOP_DATA_U32:
    case TOPSOP_DATA_I64:
    case TOPSOP_DATA_U64:
    case TOPSOP_DATA_PRED:
    case TOPSOP_DATA_I4:
      ss << scalar_value.ival;
      break;
    case TOPSOP_DATA_FP16:
    case TOPSOP_DATA_BF16:
    case TOPSOP_DATA_FP32:
    case TOPSOP_DATA_EF32:
    case TOPSOP_DATA_TF32:
    case TOPSOP_DATA_F64:
      ss << scalar_value.fval;
      break;
    default: {
      ss << "UnknownValue";
      break;
    }
  }
  ss << ")>";
  return ss.str();
}

}  // namespace custom_kernel
