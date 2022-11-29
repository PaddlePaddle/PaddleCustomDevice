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

namespace custom_kernel {
namespace experimental {

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

}  // namespace experimental
}  // namespace custom_kernel
