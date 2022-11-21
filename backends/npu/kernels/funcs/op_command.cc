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

#include "kernels/funcs/op_command.h"

#include "acl/acl_op_compiler.h"
#include "kernels/funcs/npu_funcs.h"
#include "pybind11/pybind11.h"

#define RELEASE_GIL_THEN_RUN(expr)        \
  if (PyGILState_Check()) {               \
    pybind11::gil_scoped_release release; \
    expr;                                 \
  } else {                                \
    expr;                                 \
  }

namespace custom_kernel {
namespace experimental {

aclDataType ConvertToNpuDtype(paddle::experimental::DataType dtype) {
  static std::map<paddle::experimental::DataType, aclDataType>  //
      DTYPE_2_ACL_DTYPE = {
          {paddle::experimental::DataType::BOOL, ACL_BOOL},
          {paddle::experimental::DataType::UINT8, ACL_UINT8},
          {paddle::experimental::DataType::INT8, ACL_INT8},
          {paddle::experimental::DataType::INT16, ACL_INT16},
          {paddle::experimental::DataType::INT32, ACL_INT32},
          {paddle::experimental::DataType::INT64, ACL_INT64},
          {paddle::experimental::DataType::FLOAT16, ACL_FLOAT16},
          {paddle::experimental::DataType::FLOAT32, ACL_FLOAT},
          {paddle::experimental::DataType::FLOAT64, ACL_DOUBLE},
      };
  auto iter = DTYPE_2_ACL_DTYPE.find(dtype);
  PADDLE_ENFORCE_NE(
      iter,
      DTYPE_2_ACL_DTYPE.end(),
      phi::errors::NotFound(
          "The data type %s can not convert to ACL data type.", dtype));
  return iter->second;
}

ge::DataType ConvertToGEDtype(paddle::experimental::DataType dtype) {
  static std::map<paddle::experimental::DataType, ge::DataType>  //
      DTYPE_2_GE_DTYPE = {
          {paddle::experimental::DataType::BOOL, ge::DataType::DT_UINT8},
          {paddle::experimental::DataType::UINT8, ge::DataType::DT_UINT8},
          {paddle::experimental::DataType::INT8, ge::DataType::DT_INT8},
          {paddle::experimental::DataType::INT16, ge::DataType::DT_INT16},
          {paddle::experimental::DataType::INT32, ge::DataType::DT_INT32},
          {paddle::experimental::DataType::INT64, ge::DataType::DT_INT64},
          {paddle::experimental::DataType::FLOAT16, ge::DataType::DT_FLOAT16},
          {paddle::experimental::DataType::FLOAT32, ge::DataType::DT_FLOAT},
          {paddle::experimental::DataType::FLOAT64, ge::DataType::DT_DOUBLE},
      };
  auto iter = DTYPE_2_GE_DTYPE.find(dtype);
  PADDLE_ENFORCE_NE(
      iter,
      DTYPE_2_GE_DTYPE.end(),
      phi::errors::NotFound(
          "The data type %s can not convert to ACL data type.", dtype));
  return iter->second;
}

aclFormat ConvertToNpuFormat(phi::DataLayout layout) {
  static std::map<phi::DataLayout, aclFormat> DATA_LAYOUT_2_ACL_FORMAT = {
      {phi::DataLayout::NCHW, ACL_FORMAT_NCHW},
      {phi::DataLayout::NHWC, ACL_FORMAT_NHWC},
      {phi::DataLayout::kNCDHW, ACL_FORMAT_NCDHW},
      {phi::DataLayout::kNDHWC, ACL_FORMAT_NDHWC},
      {phi::DataLayout::ANY, ACL_FORMAT_ND},
  };
  auto iter = DATA_LAYOUT_2_ACL_FORMAT.find(layout);
  PADDLE_ENFORCE_NE(
      iter,
      DATA_LAYOUT_2_ACL_FORMAT.end(),
      phi::errors::NotFound(
          "The data type (%s) can not convert to ACL data type.", layout));
  return iter->second;
}

ge::Format ConvertToGEFormat(phi::DataLayout layout) {
  static std::map<phi::DataLayout, ge::Format> DATA_LAYOUT_2_GE_FORMAT = {
      {phi::DataLayout::NCHW, ge::Format::FORMAT_NCHW},
      {phi::DataLayout::NHWC, ge::Format::FORMAT_NHWC},
      {phi::DataLayout::kNCDHW, ge::Format::FORMAT_NCDHW},
      {phi::DataLayout::kNDHWC, ge::Format::FORMAT_NDHWC},
      {phi::DataLayout::ANY, ge::Format::FORMAT_ND},
  };
  auto iter = DATA_LAYOUT_2_GE_FORMAT.find(layout);
  PADDLE_ENFORCE_NE(
      iter,
      DATA_LAYOUT_2_GE_FORMAT.end(),
      phi::errors::NotFound(
          "The data type (%s) can not convert to ACL data type.", layout));
  return iter->second;
}

class AclCommandHelper {
 public:
  static aclTensorDesc *CreateDesc(TensorDescMaker maker) {
    auto dims = phi::vectorize(maker.dims_);
    if (dims.size() == 0) {
      maker.MarkAsScalar();
    }

    if (maker.is_scalar_) {
      maker.layout_ = phi::DataLayout::ANY;
    }

    auto desc = aclCreateTensorDesc(ConvertToNpuDtype(maker.dtype_),
                                    maker.is_scalar_ ? 0 : dims.size(),
                                    maker.is_scalar_ ? nullptr : dims.data(),
                                    ConvertToNpuFormat(maker.layout_));
    PADDLE_ENFORCE_NOT_NULL(
        desc, phi::errors::External("Call aclCreateTensorDesc failed."));

    if (maker.is_host_) {
      PADDLE_ENFORCE_NPU_SUCCESS(aclSetTensorPlaceMent(desc, ACL_MEMTYPE_HOST));
    }

    if (maker.change_storage_) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclSetTensorFormat(desc, ConvertToNpuFormat(maker.layout_)));
      if (dims.size()) {
        PADDLE_ENFORCE_NPU_SUCCESS(
            aclSetTensorShape(desc,
                              maker.is_scalar_ ? 0 : dims.size(),
                              maker.is_scalar_ ? nullptr : dims.data()));
      }
    }
    return desc;
  }

  static aclDataBuffer *CreateBuffer(const void *data, size_t numel) {
    return aclCreateDataBuffer(const_cast<void *>(data), numel);
  }

  static void ConvertScalarToDeviceTensor(const phi::CustomContext &ctx,
                                          const phi::Scalar &scalar,
                                          phi::DenseTensor *tensor) {
    tensor->Resize({1});
    if (scalar.dtype() == phi::DataType::FLOAT16) {
      auto val = scalar.to<phi::dtype::float16>();
      ctx.template Alloc<phi::dtype::float16>(tensor);
      AsyncMemCpyH2D(nullptr,
                     reinterpret_cast<C_Stream>(ctx.stream()),
                     tensor->data(),
                     &val,
                     sizeof(phi::dtype::float16));
    } else if (scalar.dtype() == phi::DataType::FLOAT32) {
      auto val = scalar.to<float>();
      ctx.template Alloc<float>(tensor);
      AsyncMemCpyH2D(nullptr,
                     reinterpret_cast<C_Stream>(ctx.stream()),
                     tensor->data(),
                     &val,
                     sizeof(float));
    } else if (scalar.dtype() == phi::DataType::FLOAT64) {
      auto val = scalar.to<double>();
      ctx.template Alloc<double>(tensor);
      AsyncMemCpyH2D(nullptr,
                     reinterpret_cast<C_Stream>(ctx.stream()),
                     tensor->data(),
                     &val,
                     sizeof(double));
    } else if (scalar.dtype() == phi::DataType::INT8) {
      auto val = scalar.to<int8_t>();
      ctx.template Alloc<int8_t>(tensor);
      AsyncMemCpyH2D(nullptr,
                     reinterpret_cast<C_Stream>(ctx.stream()),
                     tensor->data(),
                     &val,
                     sizeof(int8_t));
    } else if (scalar.dtype() == phi::DataType::INT16) {
      auto val = scalar.to<int16_t>();
      ctx.template Alloc<int16_t>(tensor);
      AsyncMemCpyH2D(nullptr,
                     reinterpret_cast<C_Stream>(ctx.stream()),
                     tensor->data(),
                     &val,
                     sizeof(int16_t));
    } else if (scalar.dtype() == phi::DataType::INT32) {
      auto val = scalar.to<int32_t>();
      ctx.template Alloc<int32_t>(tensor);
      AsyncMemCpyH2D(nullptr,
                     reinterpret_cast<C_Stream>(ctx.stream()),
                     tensor->data(),
                     &val,
                     sizeof(int32_t));
    } else if (scalar.dtype() == phi::DataType::INT64) {
      auto val = scalar.to<int64_t>();
      ctx.template Alloc<int64_t>(tensor);
      AsyncMemCpyH2D(nullptr,
                     reinterpret_cast<C_Stream>(ctx.stream()),
                     tensor->data(),
                     &val,
                     sizeof(int64_t));
    } else if (scalar.dtype() == phi::DataType::BOOL) {
      uint8_t val = static_cast<uint8_t>(scalar.to<bool>());
      ctx.template Alloc<uint8_t>(tensor);
      AsyncMemCpyH2D(nullptr,
                     reinterpret_cast<C_Stream>(ctx.stream()),
                     tensor->data(),
                     &val,
                     sizeof(uint8_t));
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert data type %d scalar to tensor", scalar.dtype()));
    }
  }
};

class AclCommand : public NpuCommand {
 public:
  explicit AclCommand(const std::string &op_type) : NpuCommand(op_type) {}

  ~AclCommand() override {
    for (auto &t : in_descs_) {
      aclDestroyTensorDesc(t);
    }

    for (auto &t : in_buffers_) {
      aclDestroyDataBuffer(t);
    }

    for (auto &t : out_descs_) {
      aclDestroyTensorDesc(t);
    }

    for (auto &t : out_buffers_) {
      aclDestroyDataBuffer(t);
    }
  }

  void AddInput() override {
    VLOG(4) << "NoneInput dtype:: None";
    auto desc =
        aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    auto buffer = AclCommandHelper::CreateBuffer(nullptr, 0);
    in_descs_.push_back(desc);
    in_buffers_.push_back(buffer);
  }

  void AddInput(const phi::Scalar &tensor) override {
    VLOG(4) << "ScalarInput dtype:" << ConvertToNpuDtype(tensor.dtype()) << " "
            << "rank:" << 1 << " dims: "
            << "scalar"
            << " format:" << ACL_FORMAT_ND;
    placeholder_storage_.push_back(in_buffers_.size());
    scalar_storage_.push_back(tensor);
    storage_.emplace_back();

    in_buffers_.push_back(nullptr);
    in_descs_.push_back(nullptr);
  }

  void AddInput(const phi::DenseTensor &tensor) override {
    AddInput(tensor, TensorDescMaker("", tensor).ChangeStorage());
  }

  void AddInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    VLOG(4) << "Input dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "NPU ptr: " << tensor.data() << ", size: " << tensor.capacity();

    auto desc = AclCommandHelper::CreateDesc(maker);
    auto buffer =
        AclCommandHelper::CreateBuffer(tensor.data(), tensor.capacity());

    in_descs_.push_back(desc);
    in_buffers_.push_back(buffer);
  }

  void AddScalarInput(const phi::DenseTensor &tensor) override {
    AddScalarInput(tensor,
                   TensorDescMaker("", tensor)
                       .MarkAsScalar()
                       .SetDataLayout(phi::DataLayout::ANY));
  }

  void AddScalarInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    VLOG(4) << "ScalarInput dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "NPU ptr: " << tensor.data() << ", size: " << tensor.capacity();

    auto desc = AclCommandHelper::CreateDesc(maker);
    auto buffer =
        AclCommandHelper::CreateBuffer(tensor.data(), tensor.capacity());

    in_descs_.push_back(desc);
    in_buffers_.push_back(buffer);
  }

  void AddHostInput(const phi::DenseTensor &tensor) override {
    AddHostInput(tensor, TensorDescMaker("", tensor).MarkAsHost());
  }

  void AddHostInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    VLOG(4) << "HostInput dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "NPU ptr: " << tensor.data() << ", size: " << tensor.capacity();

    auto desc = AclCommandHelper::CreateDesc(maker);
    auto buffer =
        AclCommandHelper::CreateBuffer(tensor.data(), tensor.capacity());

    in_descs_.push_back(desc);
    in_buffers_.push_back(buffer);
  }

  void AddHostScalarInput(const phi::DenseTensor &tensor) override {
    AddHostScalarInput(tensor, TensorDescMaker("", tensor));
  }

  void AddHostScalarInput(const phi::DenseTensor &tensor,
                          TensorDescMaker maker) {
    AddScalarInput(
        tensor,
        maker.MarkAsScalar().MarkAsHost().SetDataLayout(phi::DataLayout::ANY));
  }

  void AddOutput() override {
    VLOG(4) << "NoneOutput dtype:: None";
    auto desc =
        aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    auto buffer = AclCommandHelper::CreateBuffer(nullptr, 0);
    out_descs_.push_back(desc);
    out_buffers_.push_back(buffer);
  }

  void AddOutput(phi::DenseTensor &tensor) override {
    AddOutput(tensor, TensorDescMaker("", tensor).ChangeStorage());
  }

  void AddOutput(phi::DenseTensor &tensor, TensorDescMaker maker) {
    VLOG(4) << "Output dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "NPU ptr: " << tensor.data() << ", size: " << tensor.capacity();

    auto desc = AclCommandHelper::CreateDesc(maker);
    auto buffer =
        AclCommandHelper::CreateBuffer(tensor.data(), tensor.capacity());

    out_descs_.push_back(desc);
    out_buffers_.push_back(buffer);
  }

  void AddAttribute(const std::string &name,
                    const NpuAttribute &attr) override {
    if (!attr_) {
      attr_ = aclopCreateAttr();
    }
    if (attr.type() == typeid(bool)) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrBool(attr_, name.c_str(), paddle::get<bool>(attr)));
    } else if (attr.type() == typeid(int)) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrInt(attr_, name.c_str(), paddle::get<int32_t>(attr)));
    } else if (attr.type() == typeid(int64_t)) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrInt(attr_, name.c_str(), paddle::get<int64_t>(attr)));
    } else if (attr.type() == typeid(float)) {
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrFloat(attr_, name.c_str(), paddle::get<float>(attr)));
    } else if (attr.type() == typeid(std::vector<bool>)) {
      auto a = paddle::get<std::vector<bool>>(attr);
      std::vector<uint8_t> cast_a;
      for (auto it : a) {
        cast_a.push_back(static_cast<uint8_t>(it));
      }
      PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListBool(
          attr_, name.c_str(), cast_a.size(), cast_a.data()));
    } else if (attr.type() == typeid(std::vector<int32_t>)) {
      auto a = paddle::get<std::vector<int32_t>>(attr);
      std::vector<int64_t> cast_a;
      for (auto it : a) {
        cast_a.push_back(static_cast<int64_t>(it));
      }
      PADDLE_ENFORCE_NPU_SUCCESS(aclopSetAttrListInt(
          attr_, name.c_str(), cast_a.size(), cast_a.data()));
    } else if (attr.type() == typeid(std::vector<int64_t>)) {
      auto a = paddle::get<std::vector<int64_t>>(attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListInt(attr_, name.c_str(), a.size(), a.data()));
    } else if (attr.type() == typeid(std::vector<float>)) {
      auto a = paddle::get<std::vector<float>>(attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListFloat(attr_, name.c_str(), a.size(), a.data()));
    } else if (attr.type() == typeid(std::string)) {
      auto a = paddle::get<std::string>(attr);
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrString(attr_, name.c_str(), a.c_str()));
    } else if (attr.type() == typeid(std::vector<std::string>)) {
      auto a = paddle::get<std::vector<std::string>>(attr);
      std::vector<const char *> s;
      for (auto &it : a) {
        s.push_back(it.data());
      }
      PADDLE_ENFORCE_NPU_SUCCESS(
          aclopSetAttrListString(attr_, name.c_str(), s.size(), s.data()));
    } else if (attr.type() == typeid(std::vector<std::vector<int64_t>>)) {
      auto a = paddle::get<std::vector<std::vector<int64_t>>>(attr);
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
  }

  void Run(const phi::CustomContext &ctx) override {
    ProcessScalarPlaceholder(ctx);

    auto stream = reinterpret_cast<C_Stream>(ctx.stream());

    PADDLE_ENFORCE_NOT_NULL(
        stream,
        phi::errors::External("Stream should not be null, please check."));

    VLOG(5) << "NpuOpRunner(" << this << ") Run:";
    VLOG(4) << "op_type: " << op_type_;
    VLOG(4) << "input_desc.size: " << in_descs_.size();
    VLOG(4) << "output_desc.size: " << out_buffers_.size();
    VLOG(4) << "attr: " << attr_;
    VLOG(4) << "stream: " << stream;
    aclError ret;

    // Ensure that the Gil has been released before running
    // aclopCompileAndExecute.
    RELEASE_GIL_THEN_RUN({
      ret = aclopCompileAndExecute(op_type_.c_str(),
                                   in_descs_.size(),
                                   in_descs_.data(),
                                   in_buffers_.data(),
                                   out_descs_.size(),
                                   out_descs_.data(),
                                   out_buffers_.data(),
                                   attr_,
                                   ACL_ENGINE_SYS,
                                   ACL_COMPILE_SYS,
                                   NULL,
                                   stream);
    });
    VLOG(4) << "after aclopCompileAndExecute: " << ret;
    //   ret = aclrtSynchronizeStream(stream);
    //   VLOG(4) << "after aclrtSynchronizeStream: " << ret;
    PADDLE_ENFORCE_NPU_SUCCESS(ret);
  }

  void ProcessScalarPlaceholder(const phi::CustomContext &ctx) {
    for (auto i = 0; i < storage_.size(); ++i) {
      auto &tensor = storage_[i];
      auto &scalar = scalar_storage_[i];
      AclCommandHelper::ConvertScalarToDeviceTensor(ctx, scalar, &tensor);
      auto desc = AclCommandHelper::CreateDesc(
          TensorDescMaker("", tensor)
              .MarkAsScalar()
              .MarkAsHost()
              .SetDataLayout(phi::DataLayout::ANY));
      auto buffer =
          AclCommandHelper::CreateBuffer(tensor.data(), tensor.capacity());
      in_descs_[placeholder_storage_[i]] = desc;
      in_buffers_[placeholder_storage_[i]] = buffer;
    }
  }

 private:
  std::vector<aclDataBuffer *> in_buffers_;
  std::vector<aclTensorDesc *> in_descs_;
  std::vector<aclDataBuffer *> out_buffers_;
  std::vector<aclTensorDesc *> out_descs_;

  std::vector<int> placeholder_storage_;
  std::vector<phi::Scalar> scalar_storage_;
  std::vector<phi::DenseTensor> storage_;
  aclopAttr *attr_{nullptr};
};

// graph command

class GraphCommandHelper {
 public:
  static C_GE_Tensor *ConvertDenseTensorToGETensor(const phi::DenseTensor &t) {
    auto r = CreateTensor();
    auto dims = phi::vectorize(t.dims());
    SetTensor(r,
              const_cast<void *>(t.data()),
              dims.data(),
              dims.size(),
              experimental::ConvertToGEDtype(t.dtype()),
              ge::Format::FORMAT_ND);
    return r;
  }

  static C_GE_Tensor *ConvertDenseTensorToGETensor(const phi::DenseTensor &t,
                                                   phi::DataLayout layout) {
    auto r = CreateTensor();
    auto dims = phi::vectorize(t.dims());
    SetTensor(r,
              const_cast<void *>(t.data()),
              dims.data(),
              dims.size(),
              experimental::ConvertToGEDtype(t.dtype()),
              experimental::ConvertToGEFormat(layout));
    return r;
  }

  static C_GE_Tensor *ConvertScalarToGETensor(const phi::Scalar &scalar) {
    auto r = CreateTensor();
    std::vector<int64_t> dims({1});
    if (scalar.dtype() == phi::DataType::FLOAT16) {
      auto data = scalar.to<phi::dtype::float16>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::FLOAT32) {
      auto data = scalar.to<float>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::FLOAT64) {
      auto data = scalar.to<double>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::INT8) {
      auto data = scalar.to<int8_t>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::INT16) {
      auto data = scalar.to<int16_t>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::INT32) {
      auto data = scalar.to<int32_t>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::INT64) {
      auto data = scalar.to<int64_t>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else if (scalar.dtype() == phi::DataType::BOOL) {
      auto data = scalar.to<bool>();
      SetTensor(r,
                reinterpret_cast<void *>(&data),
                dims.data(),
                dims.size(),
                experimental::ConvertToGEDtype(scalar.dtype()),
                ge::Format::FORMAT_ND);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert data type %d scalar to tensor", scalar.dtype()));
    }
    return r;
  }

  static void ConvertScalarToHostTensor(const phi::CustomContext &ctx,
                                        phi::Scalar &scalar,
                                        phi::DenseTensor *tensor) {
    tensor->Resize({1});
    if (scalar.dtype() == phi::DataType::FLOAT16) {
      auto data = ctx.template HostAlloc<phi::dtype::float16>(tensor);
      *data = scalar.to<phi::dtype::float16>();
    } else if (scalar.dtype() == phi::DataType::FLOAT32) {
      auto data = ctx.template HostAlloc<float>(tensor);
      *data = scalar.to<float>();
    } else if (scalar.dtype() == phi::DataType::FLOAT64) {
      auto data = ctx.template HostAlloc<double>(tensor);
      *data = scalar.to<double>();
    } else if (scalar.dtype() == phi::DataType::INT8) {
      auto data = ctx.template HostAlloc<int8_t>(tensor);
      *data = scalar.to<int8_t>();
    } else if (scalar.dtype() == phi::DataType::INT16) {
      auto data = ctx.template HostAlloc<int16_t>(tensor);
      *data = scalar.to<int16_t>();
    } else if (scalar.dtype() == phi::DataType::INT32) {
      auto data = ctx.template HostAlloc<int32_t>(tensor);
      *data = scalar.to<int32_t>();
    } else if (scalar.dtype() == phi::DataType::INT64) {
      auto data = ctx.template HostAlloc<int64_t>(tensor);
      *data = scalar.to<int64_t>();
    } else if (scalar.dtype() == phi::DataType::BOOL) {
      auto data = ctx.template HostAlloc<uint8_t>(tensor);
      *data = static_cast<uint8_t>(scalar.to<bool>());
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert data type %d scalar to tensor", scalar.dtype()));
    }
  }

  static std::shared_ptr<OpNode> ConvertHostTensorToConstOp(
      const phi::DenseTensor &t) {
    auto ge_tensor = GraphCommandHelper::ConvertDenseTensorToGETensor(t);
    auto const_op = std::make_shared<OpNode>("Const");
    OperatorSetAttrTensor(const_op->ge_op_, "value", ge_tensor);
    DestroyTensor(ge_tensor);

    auto dims = phi::vectorize(t.dims());
    OperatorUpdateOutputDesc(
        const_op->ge_op_,
        "y",
        dims.data(),
        dims.size(),
        custom_kernel::experimental::ConvertToGEDtype(t.dtype()),
        ge::Format::FORMAT_ND);
    return std::move(const_op);
  }

  static std::shared_ptr<OpNode> ConvertHostTensorToConstOp(
      const phi::DenseTensor &t,
      const std::vector<int64_t> &dims,
      phi::DataType dtype,
      phi::DataLayout format) {
    auto ge_tensor = GraphCommandHelper::ConvertDenseTensorToGETensor(t);
    auto const_op = std::make_shared<OpNode>("Const");
    OperatorSetAttrTensor(const_op->ge_op_, "value", ge_tensor);
    DestroyTensor(ge_tensor);

    OperatorUpdateOutputDesc(
        const_op->ge_op_,
        "y",
        const_cast<int64_t *>(dims.data()),
        dims.size(),
        custom_kernel::experimental::ConvertToGEDtype(dtype),
        custom_kernel::experimental::ConvertToGEFormat(format));
    return std::move(const_op);
  }

  static std::shared_ptr<OpNode> ConvertScalarToConstOp(const phi::Scalar &t) {
    auto ge_tensor = GraphCommandHelper::ConvertScalarToGETensor(t);
    auto const_op = std::make_shared<OpNode>("Const");
    OperatorSetAttrTensor(const_op->ge_op_, "value", ge_tensor);
    DestroyTensor(ge_tensor);

    std::vector<int64_t> dims({1});
    OperatorUpdateOutputDesc(
        const_op->ge_op_,
        "y",
        dims.data(),
        dims.size(),
        custom_kernel::experimental::ConvertToGEDtype(t.dtype()),
        ge::Format::FORMAT_ND);
    return std::move(const_op);
  }
};

class GraphCommand : public NpuCommand {
 public:
  explicit GraphCommand(const std::string &op_type)
      : NpuCommand(op_type), op_type_(op_type) {
    node_ = std::make_shared<OpNode>(op_type);
  }

  ~GraphCommand() override {}

  void AddInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    auto ge_tensor = const_cast<TensorNode *>(
        reinterpret_cast<const TensorNode *>(tensor.data()));

    VLOG(4) << "Input dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << ge_tensor;

    if (!ge_tensor->WithoutNode()) {
      OperatorSetInput(node_->ge_op_,
                       in_index_,
                       ge_tensor->in_->ge_op_,
                       ge_tensor->in_index_);
    } else {
      ge_tensor->SetTag(TensorNodeTag::UNDEFINED);
    }

    node_->PushInput(ge_tensor);
    ge_tensor->PushOutput(node_, in_index_);
    in_index_++;

    if (maker.Valid()) {
      auto dims = phi::vectorize(maker.dims_);
      OperatorUpdateInputDesc(node_->ge_op_,
                              maker.desc_name_.c_str(),
                              dims.data(),
                              dims.size(),
                              ConvertToGEDtype(maker.dtype_),
                              ConvertToGEFormat(maker.layout_));
    }
  }

  void AddHostInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    auto ge_tensor = TensorNode::malloc();

    VLOG(4) << "HostInput dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << ge_tensor;

    ge_tensor->SetInput(GraphCommandHelper::ConvertHostTensorToConstOp(tensor),
                        0);
    ge_tensor->PushOutput(node_, in_index_);
    OperatorSetInput(
        node_->ge_op_, in_index_, ge_tensor->in_->ge_op_, ge_tensor->in_index_);
    node_->PushInput(ge_tensor);
    in_index_++;

    if (maker.Valid()) {
      auto dims = phi::vectorize(tensor.dims());
      OperatorUpdateInputDesc(node_->ge_op_,
                              maker.desc_name_.c_str(),
                              dims.data(),
                              dims.size(),
                              ConvertToGEDtype(maker.dtype_),
                              ConvertToGEFormat(maker.layout_));
    }
  }

  void AddScalarInput(const phi::DenseTensor &tensor, TensorDescMaker maker) {
    auto ge_tensor = TensorNode::malloc();
    VLOG(4) << "ScalarInput dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << ge_tensor;

    ge_tensor->SetInput(GraphCommandHelper::ConvertHostTensorToConstOp(
                            tensor, {}, tensor.dtype(), phi::DataLayout::ANY),
                        0);
    ge_tensor->PushOutput(node_, in_index_);
    OperatorSetInput(
        node_->ge_op_, in_index_, ge_tensor->in_->ge_op_, ge_tensor->in_index_);
    node_->PushInput(ge_tensor);
    in_index_++;

    if (maker.Valid()) {
      auto dims = phi::vectorize(tensor.dims());
      OperatorUpdateInputDesc(node_->ge_op_,
                              maker.desc_name_.c_str(),
                              dims.data(),
                              dims.size(),
                              ConvertToGEDtype(maker.dtype_),
                              ConvertToGEFormat(maker.layout_));
    }
  }

  void AddHostScalarInput(const phi::DenseTensor &tensor,
                          TensorDescMaker maker) {
    auto ge_tensor = TensorNode::malloc();
    VLOG(4) << "HostScalarInput dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << ge_tensor;

    ge_tensor->SetInput(GraphCommandHelper::ConvertHostTensorToConstOp(
                            tensor, {}, tensor.dtype(), phi::DataLayout::ANY),
                        0);
    ge_tensor->PushOutput(node_, in_index_);
    OperatorSetInput(
        node_->ge_op_, in_index_, ge_tensor->in_->ge_op_, ge_tensor->in_index_);
    node_->PushInput(ge_tensor);
    in_index_++;

    if (maker.Valid()) {
      auto dims = phi::vectorize(tensor.dims());
      OperatorUpdateInputDesc(node_->ge_op_,
                              maker.desc_name_.c_str(),
                              dims.data(),
                              dims.size(),
                              ConvertToGEDtype(maker.dtype_),
                              ConvertToGEFormat(maker.layout_));
    }
  }

  void AddOutput(phi::DenseTensor &tensor, TensorDescMaker maker) {
    auto ge_tensor = reinterpret_cast<TensorNode *>(tensor.data());
    VLOG(4) << "Output dtype:" << maker.dtype_
            << ", rank: " << maker.dims_.size() << ", dims: " << maker.dims_
            << ", format: " << maker.layout_;
    VLOG(4) << "ptr=" << ge_tensor;

    ge_tensor->SetInput(node_, out_index_++);

    if (WithoutInput()) {
      ge_tensor->SetTag(TensorNodeTag::IN);
      ge_tensor->Cache();
    } else {
      ge_tensor->SetTag(TensorNodeTag::EDGE);
    }

    if (maker.Valid()) {
      auto dims = phi::vectorize(maker.dims_);
      OperatorUpdateOutputDesc(node_->ge_op_,
                               maker.desc_name_.c_str(),
                               dims.data(),
                               dims.size(),
                               ConvertToGEDtype(maker.dtype_),
                               ConvertToGEFormat(maker.layout_));
    }
  }

  void AddInput() override {
    VLOG(4) << "Input dtype: None";
    in_index_++;
  }

  void AddInput(const phi::DenseTensor &tensor) override {
    AddInput(tensor, TensorDescMaker(""));
  }

  void AddInput(const phi::Scalar &scalar) override {
    auto ge_tensor = TensorNode::malloc();

    VLOG(4) << "ScalarInput dtype:" << scalar.dtype();
    VLOG(4) << "ptr=" << ge_tensor;

    ge_tensor->SetInput(GraphCommandHelper::ConvertScalarToConstOp(scalar), 0);
    ge_tensor->PushOutput(node_, in_index_);
    OperatorSetInput(
        node_->ge_op_, in_index_, ge_tensor->in_->ge_op_, ge_tensor->in_index_);
    node_->PushInput(ge_tensor);
    in_index_++;
  }

  void AddHostInput(const phi::DenseTensor &tensor) override {
    AddHostInput(tensor, TensorDescMaker(""));
  }

  void AddScalarInput(const phi::DenseTensor &tensor) override {
    AddScalarInput(tensor, TensorDescMaker(""));
  }

  void AddHostScalarInput(const phi::DenseTensor &tensor) override {
    AddHostScalarInput(tensor, TensorDescMaker(""));
  }

  void AddOutput() override {
    VLOG(4) << "Output dtype: None";
    out_index_++;
  }

  void AddOutput(phi::DenseTensor &tensor) override {
    AddOutput(tensor, TensorDescMaker(""));
  }

  void AddAttribute(const std::string &name,
                    const NpuAttribute &attr) override {
    if (attr.type() == typeid(bool)) {
      OperatorSetAttrBool(node_->ge_op_, name.c_str(), paddle::get<bool>(attr));
    } else if (attr.type() == typeid(int)) {
      OperatorSetAttrInt32(
          node_->ge_op_, name.c_str(), paddle::get<int32_t>(attr));
    } else if (attr.type() == typeid(int64_t)) {
      OperatorSetAttrInt64(
          node_->ge_op_, name.c_str(), paddle::get<int64_t>(attr));
    } else if (attr.type() == typeid(float)) {
      OperatorSetAttrFloat(
          node_->ge_op_, name.c_str(), paddle::get<float>(attr));
    } else if (attr.type() == typeid(std::vector<bool>)) {
      auto a = paddle::get<std::vector<bool>>(attr);
      std::vector<uint8_t> cast_a;
      for (auto it : a) {
        cast_a.push_back(static_cast<uint8_t>(it));
      }
      OperatorSetAttrBoolList(
          node_->ge_op_, name.c_str(), cast_a.data(), cast_a.size());
    } else if (attr.type() == typeid(std::vector<int32_t>)) {
      auto a = paddle::get<std::vector<int32_t>>(attr);
      OperatorSetAttrInt32List(node_->ge_op_, name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::vector<int64_t>)) {
      auto a = paddle::get<std::vector<int64_t>>(attr);
      OperatorSetAttrInt64List(node_->ge_op_, name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::vector<float>)) {
      auto a = paddle::get<std::vector<float>>(attr);
      OperatorSetAttrFloatList(node_->ge_op_, name.c_str(), a.data(), a.size());
    } else if (attr.type() == typeid(std::string)) {
      auto a = paddle::get<std::string>(attr);
      OperatorSetAttrString(node_->ge_op_, name.c_str(), a.data());
    } else if (attr.type() == typeid(std::vector<std::string>)) {
      auto a = paddle::get<std::vector<std::string>>(attr);
      std::vector<const char *> s;
      for (auto &it : a) {
        s.push_back(it.data());
      }
      OperatorSetAttrStringList(
          node_->ge_op_, name.c_str(), s.data(), s.size());
    } else if (attr.type() == typeid(phi::DenseTensor)) {
      auto &a = paddle::get<phi::DenseTensor>(attr);
      auto cast_a = GraphCommandHelper::ConvertDenseTensorToGETensor(a);
      OperatorSetAttrTensor(node_->ge_op_, name.c_str(), cast_a);
      DestroyTensor(cast_a);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Can not convert attribubte '%s' to convert to OperatorAttr", name));
    }
  }

  void Run(const phi::CustomContext &dev_ctx) override {
    VLOG(10) << "op_type: " << op_type_;
  }

 private:
  bool WithoutInput() const { return in_index_ == 0; }

 private:
  std::shared_ptr<OpNode> node_{nullptr};
  size_t in_index_{0};
  size_t out_index_{0};

  std::vector<std::tuple<size_t, phi::Scalar *>> scalar_inputs_;
  std::string op_type_;
};

OpCommand::OpCommand(const std::string &op_type) {
  ACL_RUN(cmd_ = std::make_shared<AclCommand>(op_type));
  GRAPH_RUN(cmd_ = std::make_shared<GraphCommand>(op_type));
}

OpCommand &OpCommand::Input(const phi::Scalar &scalar) {
  cmd_->AddInput(scalar);
  return *this;
}

OpCommand &OpCommand::ScalarInput(const phi::DenseTensor &tensor) {
  if (tensor.place() == phi::CPUPlace()) {
    cmd_->AddHostScalarInput(tensor);
  } else {
    cmd_->AddScalarInput(tensor);
  }
  return *this;
}

OpCommand &OpCommand::Input(const phi::DenseTensor &tensor) {
  if (tensor.place() == phi::CPUPlace()) {
    cmd_->AddHostInput(tensor);
  } else {
    cmd_->AddInput(tensor);
  }
  return *this;
}

OpCommand &OpCommand::Output(phi::DenseTensor &tensor) {
  cmd_->AddOutput(tensor);
  return *this;
}

OpCommand &OpCommand::Input(const phi::DenseTensor &tensor,
                            TensorDescMaker maker) {
  if (tensor.place() == phi::CPUPlace()) {
    cmd_->AddHostInput(tensor, maker);
  } else {
    cmd_->AddInput(tensor, maker);
  }
  return *this;
}

OpCommand &OpCommand::ScalarInput(const phi::DenseTensor &tensor,
                                  TensorDescMaker maker) {
  if (tensor.place() == phi::CPUPlace()) {
    cmd_->AddHostScalarInput(tensor, maker);
  } else {
    cmd_->AddScalarInput(tensor, maker);
  }
  return *this;
}

OpCommand &OpCommand::Output(phi::DenseTensor &tensor, TensorDescMaker maker) {
  cmd_->AddOutput(tensor, maker);
  return *this;
}

OpCommand &OpCommand::Input() {
  cmd_->AddInput();
  return *this;
}
OpCommand &OpCommand::Output() {
  cmd_->AddOutput();
  return *this;
}

void OpCommandHelper::Assign(const phi::CustomContext &ctx,
                             const phi::DenseTensor &src,
                             phi::DenseTensor *dst) {
  ACL_RUN({ custom_kernel::TensorCopy(ctx, src, false, dst); });
  GRAPH_RUN({
    *reinterpret_cast<TensorNode *>(dst->data()) =
        *reinterpret_cast<const TensorNode *>(src.data());
  });
}

}  // namespace experimental
}  // namespace custom_kernel

struct GEGraph {
  size_t graph_id;
  C_GE_Graph *graph;
};

std::unordered_map<C_Graph, GEGraph> graph_cache;
bool graph_cache_hit = true;

C_Status graph_engine_prepare_graph(const C_Device device,
                                    const C_Stream stream,
                                    const C_Scope c_scope,
                                    const C_Graph c_graph) {
  auto session = GetSession(c_scope);
  if (graph_cache.find(c_graph) == graph_cache.end()) {
    std::string graph_name = "graph_" + std::to_string(graph_cache.size());
    auto graph_id = graph_cache.size();
    graph_cache[c_graph] = {graph_id, CreateGraph(c_scope, graph_name.c_str())};
    graph_cache_hit = false;
  } else {
    graph_cache_hit = true;
  }
}

C_Status graph_engine_execute_graph(const C_Device device,
                                    const C_Stream stream,
                                    const C_Scope c_scope,
                                    const C_Graph c_graph,
                                    char **feed_tensor_name,
                                    void **feed_tensor_data,
                                    size_t feed_tensor_num,
                                    char **fetch_tensor_name,
                                    void **fetch_tensor_data,
                                    size_t fetch_tensor_num) {
  auto session = GetSession(c_scope);

  std::vector<C_GE_Tensor *> input_tensors;
  std::vector<C_GE_Tensor *> output_tensors;

  if (!graph_cache_hit) {
    std::vector<C_GE_Operator *> input_ops;
    std::vector<C_GE_Operator *> output_ops;
    std::vector<size_t> output_ops_index;
    std::vector<C_GE_Operator *> target_ops;

    for (auto i = 0; i < feed_tensor_num; ++i) {
      auto dense_tensor =
          reinterpret_cast<phi::DenseTensor *>(feed_tensor_name[i]);
      auto graph_tensor =
          reinterpret_cast<custom_kernel::experimental::TensorNode *>(
              dense_tensor->data());

      // NOTE(wangran16): run graph failed when the Data node not be linked to
      // other node.
      if (graph_tensor->outs_.size()) {
        auto node =
            std::make_shared<custom_kernel::experimental::OpNode>("Data");
        graph_tensor->SetInput(node, 0);
        for (auto &t : graph_tensor->outs_) {
          OperatorSetInput(t.second->ge_op_,
                           t.first,
                           graph_tensor->in_->ge_op_,
                           graph_tensor->in_index_);
        }
        graph_tensor->SetTag(custom_kernel::experimental::TensorNodeTag::IN);
        graph_tensor->Cache();
      }
    }

    for (auto graph_tensor :
         custom_kernel::experimental::TensorNode::storage()) {
      if (graph_tensor->IsInput()) {
        VLOG(10) << "add in_ops: " << graph_tensor->in_->ge_op_;
        input_ops.push_back(graph_tensor->in_->ge_op_);
      }
    }

    for (auto i = 0; i < fetch_tensor_num; ++i) {
      auto dense_tensor =
          reinterpret_cast<phi::DenseTensor *>(fetch_tensor_name[i]);
      auto &graph_tensor =
          *reinterpret_cast<custom_kernel::experimental::TensorNode *>(
              dense_tensor->data());
      VLOG(10) << "add out_ops: " << graph_tensor.in_->ge_op_;
      output_ops.push_back(graph_tensor.in_->ge_op_);
      output_ops_index.push_back(graph_tensor.in_index_);
    }

    std::cerr << "input_ops size=" << input_ops.size() << std::endl;
    std::cerr << "output_ops size=" << output_ops.size() << std::endl;
    if (input_ops.size() == 0) {
      return C_FAILED;
    }
    VLOG(10) << "Set inputs: ";
    for (auto in : input_ops) {
      VLOG(10) << "in: " << in << ", type: " << OperatorGetOpType(in);
    }
    GraphSetInput(
        graph_cache[c_graph].graph, input_ops.data(), input_ops.size());
    for (auto in : output_ops) {
      VLOG(10) << "out: " << in << ", type: " << OperatorGetOpType(in);
    }
    GraphSetOutput(graph_cache[c_graph].graph,
                   output_ops.data(),
                   output_ops_index.data(),
                   output_ops.size());
    GraphSetTarget(
        graph_cache[c_graph].graph, target_ops.data(), target_ops.size());
    SessionAddGraph(
        session, graph_cache[c_graph].graph_id, graph_cache[c_graph].graph);
  }

  for (auto i = 0; i < feed_tensor_num; ++i) {
    auto dense_tensor =
        reinterpret_cast<phi::DenseTensor *>(feed_tensor_name[i]);
    auto &graph_tensor =
        *reinterpret_cast<custom_kernel::experimental::TensorNode *>(
            dense_tensor->data());
    if (graph_tensor.outs_.size()) {
      auto dims = phi::vectorize(dense_tensor->dims());
      auto tensor = CreateTensor();
      PADDLE_ENFORCE(tensor, phi::errors::Fatal("CreateTensor failed."));
      SetTensor(
          tensor,
          feed_tensor_data[i],
          dims.data(),
          dims.size(),
          custom_kernel::experimental::ConvertToGEDtype(dense_tensor->dtype()),
          custom_kernel::experimental::ConvertToGEFormat(
              dense_tensor->layout()));
      input_tensors.push_back(tensor);
    }
  }

  for (auto i = 0; i < fetch_tensor_num; ++i) {
    auto dense_tensor =
        reinterpret_cast<phi::DenseTensor *>(fetch_tensor_name[i]);
    auto &graph_tensor =
        *reinterpret_cast<custom_kernel::experimental::TensorNode *>(
            dense_tensor->data());
    auto tensor = CreateTensor();
    PADDLE_ENFORCE(tensor, phi::errors::Fatal("CreateTensor failed."));
    output_tensors.push_back(tensor);
  }

  VLOG(10) << "SessionRunGraph session=" << session
           << ", graph_id=" << graph_cache[c_graph].graph_id
           << ", input_tensors.size()=" << input_tensors.size();
  SessionRunGraph(session,
                  graph_cache[c_graph].graph_id,
                  input_tensors.data(),
                  input_tensors.size(),
                  output_tensors.data(),
                  output_tensors.size());

  for (auto i = 0; i < fetch_tensor_num; ++i) {
    memcpy(fetch_tensor_data[i],
           TensorGetData(output_tensors[i]),
           TensorGetSize(output_tensors[i]));
  }
  for (auto t : input_tensors) {
    VLOG(10) << "DestroyTensor: " << t;
    DestroyTensor(t);
  }
  for (auto t : output_tensors) {
    VLOG(10) << "DestroyTensor: " << t;
    DestroyTensor(t);
  }
  return C_SUCCESS;
}

C_Status graph_engine_initialize(const C_Device device, const C_Stream stream) {
  graph_initialize(device, stream);
  return C_SUCCESS;
}

C_Status graph_engine_finalize(const C_Device device, const C_Stream stream) {
  for (auto graph_tensor : custom_kernel::experimental::TensorNode::storage()) {
    // if (graph_tensor->is_feed_ == false) {
    //   VLOG(10) << "graph_engine_finalize deallocate: ptr=" << graph_tensor;
    //   delete graph_tensor;
    // }
    custom_kernel::experimental::TensorNode::storage().clear();
  }

  graph_finalize(device, stream);
  return C_SUCCESS;
}

C_Status graph_engine_allocator_allocate(const C_Device device,
                                         void **ptr,
                                         size_t byte_size) {
  *ptr = new custom_kernel::experimental::TensorNode;
  VLOG(10) << "graph_engine_allocator_allocate: ptr=" << *ptr
           << ", byte_size=" << byte_size;
  return C_SUCCESS;
}

C_Status graph_engine_allocator_deallocate(const C_Device device,
                                           void *ptr,
                                           size_t byte_size) {
  // VLOG(10) << "graph_engine_allocator_deallocate: ptr=" << ptr
  //          << ", byte_size=" << byte_size;
  // delete reinterpret_cast<custom_kernel::experimental::TensorNode *>(ptr);
  return C_SUCCESS;
}
