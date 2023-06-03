#pragma once
#include <concurrentqueue.h>

#include <deque>
#include <string>
#include <vector>

#include "runtime/runtime.h"

namespace custom_kernel {

using Tensor = phi::DenseTensor;
using Context = phi::CustomContext;

template <typename WideT, typename NarrowT>
NarrowT CheckedNarrowing(const WideT& wide) {
  NarrowT narrow = wide;
  CHECK_EQ(narrow, wide)
      << "checked narrowing failed; values not equal post-conversion";
  return narrow;
}

class MLUCnnlTensorDesc {
 public:
  MLUCnnlTensorDesc() {}

  // SE_DISALLOW_COPY_AND_ASSIGN
  MLUCnnlTensorDesc(const MLUCnnlTensorDesc& desc) = delete;
  MLUCnnlTensorDesc& operator=(const MLUCnnlTensorDesc&) = delete;

  MLUCnnlTensorDesc(MLUCnnlTensorDesc&& rhs)
  : raw_tensor_desc(rhs.raw_tensor_desc) {
    rhs.raw_tensor_desc = nullptr;
  }

  MLUCnnlTensorDesc& operator=(MLUCnnlTensorDesc&& rhs);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
      const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
      const cnnlDataType_t tensor_dtype, const cnnlTensorLayout_t layout);

  MLUCnnlTensorDesc(const int tensor_dim, const int dim_sizes[],
      const cnnlDataType_t tensor_dtype,
      int position);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
      const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
      const cnnlDataType_t tensor_dtype, const cnnlTensorLayout_t layout);

  MLUCnnlTensorDesc(const int tensor_dim, const int64_t dim_sizes[],
      const cnnlDataType_t tensor_dtype,
      int position);

  MLUCnnlTensorDesc(const Tensor& tensor,
      const cnnlTensorLayout_t layout,
      const cnnlDataType_t tensor_dtype);

  MLUCnnlTensorDesc(const Tensor& tensor,
        cnnlTensorLayout_t layout,
        const cnnlDataType_t tensor_dtype,
        int position);

  MLUCnnlTensorDesc(const Tensor& tensor,
      cnnlTensorLayout_t layout,
      const cnnlDataType_t tensor_dtype,
      int position,
      float scale);

  ~MLUCnnlTensorDesc();

  const cnnlTensorDescriptor_t get() const { return raw_tensor_desc; }

 private:
  cnnlTensorDescriptor_t raw_tensor_desc = nullptr;
};

class MLUCnnlActivationDesc {
 public:
  MLUCnnlActivationDesc(const MLUCnnlActivationDesc& desc) = delete;
  MLUCnnlActivationDesc& operator=(const MLUCnnlActivationDesc& desc) = delete;
  MLUCnnlActivationDesc(const cnnlActivationMode_t act_mode, const float ceof);

  const cnnlActivationDescriptor_t get() const;
  ~MLUCnnlActivationDesc();

 private:
  cnnlActivationDescriptor_t active_desc_ = nullptr;
};

class MLUCnnl {
 public:
  static void Active(const Context& ctx,
                     cnnlActivationDescriptor_t active_desc,
                     const cnnlTensorDescriptor_t input_desc,
                     const void* input,
                     const cnnlTensorDescriptor_t output_desc,
                     void* output);
};

}  // namespace custom_kernel
