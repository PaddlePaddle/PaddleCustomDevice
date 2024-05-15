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
#ifdef PADDLE_WITH_ATB

#include "atb/atb_infer.h"
#include "kernels/funcs/npu_funcs.h"

#define ATB_CHECK(func) RUNTIME_CHECK(func, atb::NO_ERROR)

namespace atb_layers {

template <typename T>
struct to_acl_dtype;

template <>
struct to_acl_dtype<int8_t> {
  static const aclDataType value() { return ACL_INT8; }
};

class OperationRunner {
 public:
  OperationRunner() {}

  ~OperationRunner() {
    if (operation_) {
      ATB_CHECK(atb::DestroyOperation(operation_));
    }
  }

  template <typename OpParam>
  void create(const OpParam& param) {
    ATB_CHECK(atb::CreateOperation(param, &operation_));
  }

  void bind_input(const void* src,
                  const phi::DataType& dtype,
                  const std::vector<int64_t>& dims) {
    variant_pack_.inTensors.resize(in_tensor_num_ + 1);
    atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
    atb_tensor.desc.format = ACL_FORMAT_ND;
    atb_tensor.desc.shape.dimNum = dims.size();
    for (auto i = 0; i < dims.size(); ++i) {
      atb_tensor.desc.shape.dims[i] = dims[i];
    }
    atb_tensor.desc.dtype = ConvertToNpuDtype(dtype);
    atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
    atb_tensor.hostData = nullptr;
    atb_tensor.deviceData = const_cast<void*>(src);
  }

  void bind_host_input(const void* src,
                       const phi::DataType& dtype,
                       const std::vector<int64_t>& dims) {
    variant_pack_.inTensors.resize(in_tensor_num_ + 1);
    atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
    atb_tensor.desc.format = ACL_FORMAT_ND;
    atb_tensor.desc.shape.dimNum = dims.size();
    for (auto i = 0; i < dims.size(); ++i) {
      atb_tensor.desc.shape.dims[i] = dims[i];
    }
    atb_tensor.desc.dtype = ConvertToNpuDtype(dtype);
    atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
    atb_tensor.deviceData = nullptr;
    atb_tensor.hostData = const_cast<void*>(src);
  }

  void bind_input(const void* src,
                  const void* host_src,
                  const phi::DataType& dtype,
                  const std::vector<int64_t>& dims) {
    variant_pack_.inTensors.resize(in_tensor_num_ + 1);
    atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
    atb_tensor.desc.format = ACL_FORMAT_ND;
    atb_tensor.desc.shape.dimNum = dims.size();
    for (auto i = 0; i < dims.size(); ++i) {
      atb_tensor.desc.shape.dims[i] = dims[i];
    }
    atb_tensor.desc.dtype = ConvertToNpuDtype(dtype);
    atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
    atb_tensor.hostData = const_cast<void*>(host_src);
    atb_tensor.deviceData = const_cast<void*>(src);
  }

  void bind_input(const phi::DenseTensor& src,
                  const std::vector<int64_t>& dims = {}) {
    bool is_cpu_tensor = src.place().GetType() == phi::AllocationType::CPU;
    std::vector<int64_t> new_dims;
    if (dims.size() == 0) {
      new_dims = phi::vectorize<int64_t>(src.dims());
    } else {
      new_dims = dims;
    }
    if (is_cpu_tensor) {
      bind_host_input(src.data(), src.dtype(), new_dims);
    } else {
      bind_input(src.data(), src.dtype(), new_dims);
    }
  }

  template <typename T>
  void bind_input() {
    variant_pack_.inTensors.resize(in_tensor_num_ + 1);
    atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
    atb_tensor.dataSize = 0;
    atb_tensor.hostData = nullptr;
    atb_tensor.deviceData = nullptr;
    atb_tensor.desc.format = ACL_FORMAT_ND;
    atb_tensor.desc.dtype = to_acl_dtype<T>::value();
  }

  void bind_input(const phi::DenseTensor& src,
                  const phi::DenseTensor& host_src,
                  const std::vector<int64_t>& dims = {}) {
    std::vector<int64_t> new_dims;
    if (dims.size() == 0) {
      new_dims = phi::vectorize<int64_t>(src.dims());
    } else {
      new_dims = dims;
    }
    bind_input(src.data(), host_src.data(), src.dtype(), new_dims);
  }

  void bind_output(const void* src,
                   const phi::DataType& dtype,
                   const std::vector<int64_t>& dims) {
    variant_pack_.outTensors.resize(out_tensor_num_ + 1);
    atb::Tensor& atb_tensor = variant_pack_.outTensors[out_tensor_num_++];
    atb_tensor.desc.format = ACL_FORMAT_ND;
    atb_tensor.desc.shape.dimNum = dims.size();
    for (auto i = 0; i < dims.size(); ++i) {
      atb_tensor.desc.shape.dims[i] = dims[i];
    }
    atb_tensor.desc.dtype = ConvertToNpuDtype(dtype);
    atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
    atb_tensor.hostData = nullptr;
    atb_tensor.deviceData = const_cast<void*>(src);
  }

  void bind_output(phi::DenseTensor* src,
                   const std::vector<int64_t>& dims = {}) {
    std::vector<int64_t> new_dims;
    if (dims.size() == 0) {
      new_dims = phi::vectorize<int64_t>(src->dims());
    } else {
      new_dims = dims;
    }
    bind_output(src->data(), src->dtype(), new_dims);
  }

  void bind_input(const paddle::Tensor& src,
                  const std::vector<int64_t>& dims = {}) {
    bind_input(*static_cast<const phi::DenseTensor*>(src.impl().get()), dims);
  }

  void bind_input(const paddle::Tensor& src,
                  const paddle::Tensor& host_src,
                  const std::vector<int64_t>& dims = {}) {
    bind_input(*static_cast<const phi::DenseTensor*>(src.impl().get()),
               *static_cast<const phi::DenseTensor*>(host_src.impl().get()),
               dims);
  }

  void bind_output(paddle::Tensor* src, const std::vector<int64_t>& dims = {}) {
    bind_output(static_cast<phi::DenseTensor*>(src->impl().get()), dims);
  }

  void run(const phi::CustomContext& dev_ctx) {
    auto ctx = get_context(dev_ctx);
    uint8_t* workspace = nullptr;
    uint64_t workspace_size;
    ATB_CHECK(operation_->Setup(variant_pack_, workspace_size, ctx));
    if (workspace_size > 0) {
      workspace = get_workspace(dev_ctx, workspace_size);
    }
    ATB_CHECK(
        operation_->Execute(variant_pack_, workspace, workspace_size, ctx));
  }

 private:
  atb::Context* get_context(const phi::CustomContext& dev_ctx) {
    static std::unordered_map<void*, atb::Context*> m;

    void* stream =
        const_cast<void*>(reinterpret_cast<const void*>(dev_ctx.stream()));
    if (m.find(stream) == m.end()) {
      ATB_CHECK(atb::CreateContext(&m[stream]));
      ATB_CHECK(m[stream]->SetExecuteStream(reinterpret_cast<void*>(stream)));
    }
    return m[stream];
  }

  uint8_t* get_workspace(const phi::CustomContext& dev_ctx,
                         uint64_t workspace_size) {
    static phi::DenseTensor tmp;
    if (workspace_size > tmp.numel()) {
      dev_ctx.Wait();
      tmp.Resize({workspace_size});
      dev_ctx.template Alloc<uint8_t>(&tmp);
      LOG(INFO) << "alloc workspace size: " << tmp.numel();
    }
    if (tmp.numel() == 0) {
      return nullptr;
    }
    return tmp.data<uint8_t>();
  }

 private:
  atb::Operation* operation_ = nullptr;
  atb::VariantPack variant_pack_;
  uint64_t in_tensor_num_ = 0;
  uint64_t out_tensor_num_ = 0;
};

}  // namespace atb_layers

#endif
