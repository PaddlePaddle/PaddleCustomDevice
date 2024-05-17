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
#include "kernels/funcs/op_utils.h"
#include "kernels/topsflame/include/topsop/topsop_define.h"

#define CHECK_TOPS_OP_CALL(func)                                              \
  do {                                                                        \
    auto status = (func);                                                     \
    PADDLE_ENFORCE_EQ(                                                        \
        status,                                                               \
        TOPSOP_STATUS_SUCCESS,                                                \
        phi::errors::Fatal("Failed to call tops op, get error: %d", status)); \
  } while (false)

namespace custom_kernel {
class topsopTensor {
 public:
  topsopTensor() : tops_op_tensor_handle_(nullptr) {}

  explicit topsopTensor(const topsopTensorHandle_t& tensor_handle)
      : tops_op_tensor_handle_(tensor_handle) {}

  topsopTensor(topsopSize_t* dims,
               topsopSize_t* strides,
               topsopDataType_t dtype,
               topsopDeviceMemHandle_t memhandle);

  ~topsopTensor() {
    if (tops_op_tensor_handle_ != nullptr) {
      topsopDestroyTensor(tops_op_tensor_handle_);
    }
  }

  topsopTensorHandle_t GetTensorHandle() const {
    return tops_op_tensor_handle_;
  }

  topsopDeviceMemHandle_t GetTensorData() const;

  topsopSize_t GetTensorShape() const;

  topsopSize_t GetTensorStrides() const;

  topsopDataType_t GetTensorDataType() const;

  // This function gets total elements number of the tensor.
  int64_t GetTensorElementNums() const;

  // This function gets bytes per element of the tensor.
  int64_t GetTensorBPE() const;

  std::string ToString() const;

 private:
  topsopTensorHandle_t tops_op_tensor_handle_;
};

// help func
std::ostream& operator<<(std::ostream& os, const topsopTensor& tensor);

std::ostream& operator<<(std::ostream& os, const topsopTensorHandle_t& tensor);

std::ostream& operator<<(std::ostream& os, const topsopDataType_t& dtype);

std::ostream& operator<<(std::ostream& os, const topsopStatus_t& status);

std::ostream& operator<<(std::ostream& os, const topsopSize_t& size);

std::ostream& operator<<(std::ostream& os, const topsopScalar_t& scalar);

topsopTensor CreateTopsopTensor(const phi::DenseTensor& tensor);

topsopTensorHandle_t CreateTopsopTensorHandle(const phi::DenseTensor& tensor);

topsopTensorHandle_t OptionalTensorToTopsopTensorHandle(
    const paddle::optional<phi::DenseTensor>& opt_tensor);

topsopDataType_t DataTypeToTopsopDataType(const phi::DataType& dtype);

topsopScalar_t ScalarToTopsopScalar(const phi::Scalar& scalar_value);

topsopScalar_t OptionalScalarToTopsopScalar(
    const paddle::optional<phi::Scalar>& opt_scalar);

topsopSize_t IntArrayToTopsopSize(const phi::IntArray& int_array);

topsopSize_t IntArrayToTopsopSize(const std::vector<int64_t>& int_array);

topsopSize_t OptionalIntArrayToTopsopSize(
    const paddle::optional<phi::IntArray>& opt_array);

std::string TopsopScalarToString(const topsopScalar_t& scalar_value);

}  // namespace custom_kernel
