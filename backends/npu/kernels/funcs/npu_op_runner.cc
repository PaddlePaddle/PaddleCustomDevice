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

#include "kernels/funcs/npu_op_runner.h"

#include "acl/acl_op_compiler.h"
#include "kernels/funcs/npu_enforce.h"
#include "kernels/funcs/npu_funcs.h"
#include "pybind11/pybind11.h"
#include "runtime/runtime.h"

static aclDataBuffer *float_status_buffer_ = NULL;
static aclTensorDesc *float_status_desc_ = NULL;

ENV_uint64(ascend_check_nan_inf, 0);
ENV_uint64(ascend_blocking_npu_runner, 0);

NpuOpRunner::NpuOpRunner() {}

NpuOpRunner::NpuOpRunner(const std::string &op_type) : op_type_(op_type) {}

NpuOpRunner::NpuOpRunner(const std::string &op_type,
                         const std::vector<phi::DenseTensor> &inputs,
                         const std::vector<phi::DenseTensor> &outputs,
                         const NPUAttributeMap &attrs)
    : op_type_(op_type) {
  AddInputs(inputs);
  AddOutputs(outputs);
  AddAttrs(attrs);
}

NpuOpRunner::~NpuOpRunner() {
  VLOG(4) << "Free NpuOpRunner(" << this << ") of " << op_type_;
  // Is it safe to free the descs/buffers after run called in host ?
  aclopDestroyAttr(attr_);  // return void
  for (auto desc : input_descs_) {
    aclDestroyTensorDesc(desc);
  }
  for (auto desc : output_descs_) {
    aclDestroyTensorDesc(desc);
  }
  for (auto buffer : input_buffers_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(buffer));
  }
  for (auto buffer : output_buffers_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(buffer));
  }
}

const std::string &NpuOpRunner::Type() { return op_type_; }

NpuOpRunner &NpuOpRunner::SetType(const std::string &name) {
  op_type_ = name;
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttr(const std::string &name,
                                  const NPUAttribute &attr) {
  if (!attr_) {
    attr_ = aclopCreateAttr();
  }
  if (attr.type() == typeid(bool)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrBool(attr_, name.c_str(), BOOST_GET_CONST(bool, attr)));
  } else if (attr.type() == typeid(int)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrInt(attr_, name.c_str(), BOOST_GET_CONST(int, attr)));
  } else if (attr.type() == typeid(int64_t)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrInt(attr_, name.c_str(), BOOST_GET_CONST(int64_t, attr)));
  } else if (attr.type() == typeid(float)) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrFloat(attr_, name.c_str(), BOOST_GET_CONST(float, attr)));
  } else if (attr.type() == typeid(std::vector<bool>)) {
    auto a = BOOST_GET_CONST(std::vector<bool>, attr);
    std::vector<uint8_t> cast_a;
    for (auto it : a) {
      cast_a.push_back(static_cast<uint8_t>(it));
    }
    PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListBool(
        attr_, name.c_str(), cast_a.size(), cast_a.data()));
  } else if (attr.type() == typeid(std::vector<int>)) {
    auto a = BOOST_GET_CONST(std::vector<int>, attr);
    std::vector<int64_t> cast_a;
    for (auto it : a) {
      cast_a.push_back(static_cast<int64_t>(it));
    }
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListInt(attr_, name.c_str(), cast_a.size(), cast_a.data()));
  } else if (attr.type() == typeid(std::vector<int64_t>)) {
    auto a = BOOST_GET_CONST(std::vector<int64_t>, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListInt(attr_, name.c_str(), a.size(), a.data()));
  } else if (attr.type() == typeid(std::vector<float>)) {
    auto a = BOOST_GET_CONST(std::vector<float>, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListFloat(attr_, name.c_str(), a.size(), a.data()));
  } else if (attr.type() == typeid(std::string)) {
    auto a = BOOST_GET_CONST(std::string, attr);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrString(attr_, name.c_str(), a.c_str()));
  } else if (attr.type() == typeid(std::vector<std::string>)) {
    auto a = BOOST_GET_CONST(std::vector<std::string>, attr);
    std::vector<const char *> s;
    for (auto &it : a) {
      s.push_back(it.data());
    }
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclopSetAttrListString(attr_, name.c_str(), s.size(), s.data()));
  } else if (attr.type() == typeid(std::vector<std::vector<int64_t>>)) {
    auto a = BOOST_GET_CONST(std::vector<std::vector<int64_t>>, attr);
    std::vector<int64_t *> data;
    std::vector<int> num;
    for (auto &&v : a) {
      data.push_back(v.data());
      num.push_back(v.size());
    }
    PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListListInt(
        attr_, name.c_str(), data.size(), num.data(), data.data()));
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Can not convert attribubte '%s' to convert to aclopAttr", name));
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttrDataType(const std::string &name,
                                          const NPUAttribute &attr) {
  PADDLE_ENFORCE_EQ(
      (attr.type() == typeid(int)),
      true,
      phi::errors::InvalidArgument(
          "Attr type is NOT equal to framework::proto::VarType::Type."));
  if (!attr_) {
    attr_ = aclopCreateAttr();
  }
  VLOG(4) << "AddAttrDataType call";
  auto dtype = ConvertToNpuDtype(
      static_cast<paddle::experimental::DataType>(paddle::get<int>(attr)));
  PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrDataType(attr_, name.c_str(), dtype));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttrs(const NPUAttributeMap &attrs) {
  for (const auto &pair : attrs) {
    AddAttr(pair.first, pair.second);
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(const phi::DenseTensor &tensor) {
  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(const phi::DenseTensor &tensor,
                                   aclMemType mem_type) {
  // create aclTensorDesc
  auto desc = CreateTensorDesc(tensor, mem_type);
  input_descs_.emplace_back(desc);
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(tensor));
  if (mem_type == ACL_MEMTYPE_HOST) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorConst(
        desc, const_cast<void *>(tensor.data()), tensor.capacity()));
  }

  return *this;
}

template <typename T>
NpuOpRunner &NpuOpRunner::AddInput(const phi::CustomContext &dev_ctx,
                                   const std::vector<T> &&values,
                                   const bool is_const) {
  phi::DenseTensor host_tensor;
  custom_kernel::TensorFromVector(
      dev_ctx, values, phi::CPUContext(), &host_tensor);
  host_tensors_.emplace_back(host_tensor);
  // create aclTensorDesc
  auto desc = CreateTensorDesc(host_tensor, ACL_MEMTYPE_HOST);
  input_descs_.emplace_back(desc);
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(host_tensor));
  // set tensor const
  if (is_const) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclSetTensorConst(desc, host_tensor.data(), host_tensor.capacity()));
  }
  return *this;
}

#define ADD_INPUT_IMPL_GET_DTYPE(cpp_type)               \
  template NpuOpRunner &NpuOpRunner::AddInput<cpp_type>( \
      const phi::CustomContext &dev_ctx,                 \
      const std::vector<cpp_type> &&values,              \
      const bool is_const);
ADD_INPUT_IMPL_GET_DTYPE(bool);
ADD_INPUT_IMPL_GET_DTYPE(int32_t);
ADD_INPUT_IMPL_GET_DTYPE(int64_t);
ADD_INPUT_IMPL_GET_DTYPE(float);
ADD_INPUT_IMPL_GET_DTYPE(double);
ADD_INPUT_IMPL_GET_DTYPE(phi::dtype::float16);
#undef ADD_INPUT_IMPL_GET_DTYPE

NpuOpRunner &NpuOpRunner::AddOutput(const phi::DenseTensor &tensor) {
  // create aclTensorDesc
  output_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  output_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInputs(
    const std::vector<phi::DenseTensor> &tensors) {
  input_descs_.reserve(tensors.size());
  input_buffers_.reserve(tensors.size());
  for (auto tensor : tensors) {
    // create aclTensorDesc
    input_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    input_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

// NOTE(zhiqiu): For operators whose input is a list (such as concat, stack),
// It is needed to set the name of each input tensor.
NpuOpRunner &NpuOpRunner::AddInputNames(const std::vector<std::string> &names) {
  PADDLE_ENFORCE_EQ(names.size(),
                    input_descs_.size(),
                    phi::errors::InvalidArgument(
                        "The size of input names should be "
                        "equal to the size of input descs, but got the size "
                        "of input names is %d, the size of input descs is %d.",
                        names.size(),
                        input_descs_.size()));
  for (size_t i = 0; i < names.size(); ++i) {
    aclSetTensorDescName(input_descs_[i], names[i].c_str());
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddOutputs(
    const std::vector<phi::DenseTensor> &tensors) {
  output_descs_.reserve(tensors.size());
  output_buffers_.reserve(tensors.size());
  for (auto tensor : tensors) {
    // create aclTensorDesc
    output_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    output_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

aclTensorDesc *NpuOpRunner::GetInputDesc(size_t index) {
  PADDLE_ENFORCE_LT(index,
                    input_descs_.size(),
                    phi::errors::OutOfRange(
                        "The index should be less than the size of inputs of "
                        "operator %s, but got index is %d and size is %d",
                        Type(),
                        index,
                        input_descs_.size()));
  return input_descs_[index];
}

aclTensorDesc *NpuOpRunner::GetOutputDesc(size_t index) {
  PADDLE_ENFORCE_LT(index,
                    output_descs_.size(),
                    phi::errors::OutOfRange(
                        "The index should be less than the size of output of "
                        "operator %s, but got index is %d and size is %d",
                        Type(),
                        index,
                        output_descs_.size()));
  return output_descs_[index];
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetInputDescs() {
  return input_descs_;
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetOutputDescs() {
  return output_descs_;
}

std::vector<aclDataBuffer *> &NpuOpRunner::GetInputBuffers() {
  return input_buffers_;
}

std::vector<aclDataBuffer *> &NpuOpRunner::GetOutputBuffers() {
  return output_buffers_;
}

aclTensorDesc *NpuOpRunner::CreateTensorDesc(phi::DenseTensor tensor,
                                             aclMemType mem_type) {
  auto data_type = ConvertToNpuDtype(tensor.dtype());
  auto origin_format = ConvertToNpuFormat(tensor.layout());
  auto origin_dims = phi::vectorize(tensor.dims());

  auto origin_size = origin_dims.size();
  if (op_type_ == "DropOutGenMask" && origin_size == 1 &&
      *(origin_dims.data()) == 1) {
    origin_size = 0;
  }

  auto *desc = aclCreateTensorDesc(
      data_type, origin_size, origin_dims.data(), origin_format);
  PADDLE_ENFORCE_NOT_NULL(
      desc, phi::errors::External("Call aclCreateTensorDesc failed."));

  if (tensor.storage_properties_initialized()) {
    auto npu_properties =
        tensor.storage_properties<phi::NPUStorageProperties>();
    int64_t storage_format = npu_properties.storage_format;
    auto storage_dims = phi::vectorize(npu_properties.storage_dims);
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclSetTensorFormat(desc, (aclFormat)storage_format));
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclSetTensorShape(desc, storage_dims.size(), storage_dims.data()));
    VLOG(1) << "CreateTensorDesc for OP: " << op_type_
            << ", data_type: " << data_type
            << ", origin_format: " << origin_format
            << ", storage_format: " << storage_format
            << ", origin_dims: " << tensor.dims()
            << ", storage_dims: " << npu_properties.storage_dims;
  } else {
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclSetTensorFormat(desc, (aclFormat)origin_format));
    PADDLE_ENFORCE_NPU_SUCCESS(
        aclSetTensorShape(desc, origin_size, origin_dims.data()));
    VLOG(1) << "CreateTensorDesc for OP: " << op_type_
            << ", data_type: " << data_type
            << ", origin_format: " << origin_format
            << ", storage_format: " << origin_format
            << ", origin_dims: " << tensor.dims()
            << ", storage_dims: " << tensor.dims();
  }
  if (mem_type == ACL_MEMTYPE_HOST) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorPlaceMent(desc, mem_type));
  }
  return desc;
}

aclDataBuffer *NpuOpRunner::CreateDataBuffer(phi::DenseTensor tensor) {
  void *ptr = tensor.data();
  VLOG(4) << "NPU ptr: " << ptr << ", size: " << tensor.capacity();
  auto *buffer = aclCreateDataBuffer(ptr, tensor.capacity());
  PADDLE_ENFORCE_NOT_NULL(
      buffer, phi::errors::External("Call aclCreateDataBuffer failed."));
  return buffer;
}

void NpuOpRunner::AllocFloatStatus(aclrtStream stream) const {
  std::string op_type = "NPUAllocFloatStatus";
  // Attr
  auto attr = aclopCreateAttr();
  // Execute
  aclError ret;
  if (PyGILState_Check()) {
    pybind11::gil_scoped_release release;
    ret = aclopCompileAndExecute(op_type.c_str(),
                                 0,
                                 nullptr,
                                 nullptr,
                                 1,
                                 &float_status_desc_,
                                 &float_status_buffer_,
                                 attr,
                                 ACL_ENGINE_SYS,
                                 ACL_COMPILE_SYS,
                                 NULL,
                                 stream);
  } else {
    ret = aclopCompileAndExecute(op_type.c_str(),
                                 0,
                                 nullptr,
                                 nullptr,
                                 1,
                                 &float_status_desc_,
                                 &float_status_buffer_,
                                 attr,
                                 ACL_ENGINE_SYS,
                                 ACL_COMPILE_SYS,
                                 NULL,
                                 stream);
  }
  PADDLE_ENFORCE_NPU_SUCCESS(ret);
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
  aclopDestroyAttr(attr);
}

void NpuOpRunner::ClearFloatStatus(aclrtStream stream) {
  std::string op_type = "NPUClearFloatStatus";
  // Input
  const std::vector<int64_t> dims{8};
  auto tmp_desc =
      aclCreateTensorDesc(ACL_FLOAT, dims.size(), dims.data(), ACL_FORMAT_NCHW);
  auto tmp_size = aclGetTensorDescSize(tmp_desc);
  void *tmp_ptr;
  aclrtMalloc(&tmp_ptr, tmp_size, ACL_MEM_MALLOC_NORMAL_ONLY);
  auto tmp_buffer = aclCreateDataBuffer(tmp_ptr, tmp_size);
  // Attr
  auto attr = aclopCreateAttr();
  // Execute
  aclError ret;
  if (PyGILState_Check()) {
    pybind11::gil_scoped_release release;
    ret = aclopCompileAndExecute(op_type.c_str(),
                                 1,
                                 &tmp_desc,
                                 &tmp_buffer,
                                 1,
                                 &float_status_desc_,
                                 &float_status_buffer_,
                                 attr,
                                 ACL_ENGINE_SYS,
                                 ACL_COMPILE_SYS,
                                 NULL,
                                 stream);
  } else {
    ret = aclopCompileAndExecute(op_type.c_str(),
                                 1,
                                 &tmp_desc,
                                 &tmp_buffer,
                                 1,
                                 &float_status_desc_,
                                 &float_status_buffer_,
                                 attr,
                                 ACL_ENGINE_SYS,
                                 ACL_COMPILE_SYS,
                                 NULL,
                                 stream);
  }
  PADDLE_ENFORCE_NPU_SUCCESS(ret);
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
  PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(tmp_buffer));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtFree(tmp_ptr));
  aclDestroyTensorDesc(tmp_desc);
  aclopDestroyAttr(attr);
}

void NpuOpRunner::InitFloatStatus(aclrtStream stream) const {
  // Init float_status_desc_ and float_status_buffer_ once, if
  // ascend_check_nan_inf is needed, only do ClearFloatStatus here.
  if (float_status_desc_ && float_status_buffer_) {
    if (FLAGS_ascend_check_nan_inf) {
      ClearFloatStatus(stream);
    }
    return;
  }
  // Init float_status_desc_
  const std::vector<int64_t> dims{8};
  float_status_desc_ =
      aclCreateTensorDesc(ACL_FLOAT, dims.size(), dims.data(), ACL_FORMAT_NCHW);
  auto float_status_size = aclGetTensorDescSize(float_status_desc_);
  PADDLE_ENFORCE_NOT_NULL(
      float_status_desc_,
      phi::errors::External("Call aclCreateTensorDesc failed."));
  // Init float_status_buffer
  void *float_status_ptr_;
  aclrtMalloc(&float_status_ptr_, float_status_size, ACL_MEM_MALLOC_HUGE_FIRST);
  float_status_buffer_ =
      aclCreateDataBuffer(float_status_ptr_, float_status_size);
  PADDLE_ENFORCE_NOT_NULL(
      float_status_buffer_,
      phi::errors::External("Call aclCreateDataBuffer failed."));
  // Alloc&ClearFloatStatus
  AllocFloatStatus(stream);
  ClearFloatStatus(stream);
}

void NpuOpRunner::PrintOpInfo() const {
  // PrintInput
  std::cout << "Input: " << std::endl;
  for (int i = 0; i < input_descs_.size(); i++) {
    auto type = aclGetTensorDescType(input_descs_[i]);
    std::cout << "- type: " << type << std::endl;
    auto input_size = aclGetTensorDescSize(input_descs_[i]);
    auto numel = aclGetTensorDescElementCount(input_descs_[i]);
    std::vector<float> cpu_data(numel, 0);
    auto input_ptr = aclGetDataBufferAddr(input_buffers_[i]);
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(cpu_data.data(),
                                           input_size,
                                           input_ptr,
                                           input_size,
                                           ACL_MEMCPY_DEVICE_TO_HOST));
    float sum = 0.0;
    std::cout << "- data: [";
    for (int i = 0; i < cpu_data.size(); ++i) {
      std::cout << cpu_data[i] << ",";
    }
    std::cout << "]" << std::endl;
    std::vector<float>().swap(cpu_data);
  }
  // PrintOutput
  std::cout << "Output: " << std::endl;
  for (int i = 0; i < output_descs_.size(); i++) {
    auto type = aclGetTensorDescType(output_descs_[i]);
    std::cout << "- type: " << type << std::endl;
    auto input_size = aclGetTensorDescSize(output_descs_[i]);
    auto numel = aclGetTensorDescElementCount(output_descs_[i]);
    std::vector<float> cpu_data(numel, 0);
    auto input_ptr = aclGetDataBufferAddr(output_buffers_[i]);
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(cpu_data.data(),
                                           input_size,
                                           input_ptr,
                                           input_size,
                                           ACL_MEMCPY_DEVICE_TO_HOST));
    float sum = 0.0;
    std::cout << "- data: [";
    for (int i = 0; i < cpu_data.size(); ++i) {
      std::cout << cpu_data[i] << ",";
    }
    std::cout << "]" << std::endl;
    std::vector<float>().swap(cpu_data);
  }
}
bool NpuOpRunner::GetFloatStatus(aclrtStream stream) {
  std::string op_type = "NPUGetFloatStatus";
  // Output
  const std::vector<int64_t> dims{8};
  auto tmp_desc =
      aclCreateTensorDesc(ACL_FLOAT, dims.size(), dims.data(), ACL_FORMAT_NCHW);
  auto tmp_size = aclGetTensorDescSize(tmp_desc);
  void *tmp_ptr;
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclrtMalloc(&tmp_ptr, tmp_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  auto tmp_buffer = aclCreateDataBuffer(tmp_ptr, tmp_size);
  // Attr
  auto attr = aclopCreateAttr();
  // Execute
  aclError ret;
  if (PyGILState_Check()) {
    pybind11::gil_scoped_release release;
    ret = aclopCompileAndExecute(op_type.c_str(),
                                 1,
                                 &float_status_desc_,
                                 &float_status_buffer_,
                                 1,
                                 &tmp_desc,
                                 &tmp_buffer,
                                 attr,
                                 ACL_ENGINE_SYS,
                                 ACL_COMPILE_SYS,
                                 NULL,
                                 stream);
  } else {
    ret = aclopCompileAndExecute(op_type.c_str(),
                                 1,
                                 &float_status_desc_,
                                 &float_status_buffer_,
                                 1,
                                 &tmp_desc,
                                 &tmp_buffer,
                                 attr,
                                 ACL_ENGINE_SYS,
                                 ACL_COMPILE_SYS,
                                 NULL,
                                 stream);
  }
  PADDLE_ENFORCE_NPU_SUCCESS(ret);
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
  std::vector<float> cpu_data(8, 0);
  auto float_status_size = aclGetTensorDescSize(float_status_desc_);
  auto float_status_ptr = aclGetDataBufferAddr(float_status_buffer_);
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(cpu_data.data(),
                                         float_status_size,
                                         float_status_ptr,
                                         float_status_size,
                                         ACL_MEMCPY_DEVICE_TO_HOST));
  float sum = 0.0;
  for (int i = 0; i < cpu_data.size(); ++i) {
    sum += cpu_data[i];
  }
  PADDLE_ENFORCE_NPU_SUCCESS(aclDestroyDataBuffer(tmp_buffer));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtFree(tmp_ptr));
  aclDestroyTensorDesc(tmp_desc);
  aclopDestroyAttr(attr);
  return sum >= 1.0;
}

void NpuOpRunner::Run(aclrtStream stream, bool sync) const {
  PADDLE_ENFORCE_NOT_NULL(
      stream,
      phi::errors::External("Stream should not be null, please check."));
  InitFloatStatus(stream);
  VLOG(5) << "NpuOpRunner(" << this << ") Run:";
  VLOG(4) << "op_type: " << op_type_;
  VLOG(4) << "input_desc.size: " << input_descs_.size();
  VLOG(4) << "output_desc.size: " << output_descs_.size();
  VLOG(4) << "attr: " << attr_;
  VLOG(4) << "stream: " << stream;
  aclError ret;
  // Ensure that the Gil has been released before running
  // aclopCompileAndExecute.
  if (PyGILState_Check()) {
    pybind11::gil_scoped_release release;
    ret = aclopCompileAndExecute(op_type_.c_str(),
                                 input_descs_.size(),
                                 input_descs_.data(),
                                 input_buffers_.data(),
                                 output_descs_.size(),
                                 output_descs_.data(),
                                 output_buffers_.data(),
                                 attr_,
                                 ACL_ENGINE_SYS,
                                 ACL_COMPILE_SYS,
                                 NULL,
                                 stream);
  } else {
    ret = aclopCompileAndExecute(op_type_.c_str(),
                                 input_descs_.size(),
                                 input_descs_.data(),
                                 input_buffers_.data(),
                                 output_descs_.size(),
                                 output_descs_.data(),
                                 output_buffers_.data(),
                                 attr_,
                                 ACL_ENGINE_SYS,
                                 ACL_COMPILE_SYS,
                                 NULL,
                                 stream);
  }
  VLOG(4) << "after aclopCompileAndExecute: " << ret;
  if (sync || FLAGS_ascend_blocking_npu_runner) {
    ret = aclrtSynchronizeStream(stream);
  }
  PADDLE_ENFORCE_NPU_SUCCESS(ret);
  if (FLAGS_ascend_check_nan_inf && GetFloatStatus(stream)) {
    PrintOpInfo();
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "Operator %s contains Nan/Inf.", op_type_));
  }
}
