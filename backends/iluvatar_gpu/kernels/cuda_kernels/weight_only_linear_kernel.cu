/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/weight_quantize_kernel.h"
#include "runtime/iluvatar_context.h"

namespace phi {

template <typename T, typename Context>
void WeightOnlyLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const paddle::optional<DenseTensor>& bias,
                            const DenseTensor& weight_scale,
                            const std::string& weight_dtype,
                            const int32_t arch,
                            const int32_t group_size,
                            DenseTensor* out) {
  /**
   * x:[m,k]
   * w:[n,k]
   * out:[m,n]
   * w_scale:[k/group_size, n] or [n]
   * bias:[n]
   */
  dev_ctx.template Alloc<T>(out);
  const T* x_data = x.data<T>();
  const int8_t* weight_data = weight.data<int8_t>();
  const T* bias_data = bias ? bias.get().data<T>() : nullptr;
  const T* weight_scale_data = weight_scale.data<T>();
  T* out_data = out->data<T>();
  const auto x_dims = x.dims();
  const auto w_dims = weight.dims();
  PADDLE_ENFORCE_EQ(
      ((group_size == -1)),
      true,
      common::errors::InvalidArgument("group_size only support -1."));
  int n = group_size > 0 ? weight_scale.dims()[1] : weight_scale.dims()[0];
  int k = w_dims[1];
  int m = x.numel() / k;
  PADDLE_ENFORCE_EQ(
      ((k % 4 == 0)),
      true,
      common::errors::InvalidArgument("Only support K % 4 == 0."));

  cuinferHandle_t handle = iluvatar::getContextInstance()->getIxInferHandle();
  cuinferPointerMode_t cuinfer_ptr_mode = CUINFER_POINTER_MODE_HOST;
  cuinferOperation_t transa = CUINFER_OP_T;
  cuinferOperation_t transb = CUINFER_OP_N;
  cudaDataType_t a_type;
  cudaDataType_t bc_type;
  if (weight_dtype == "int8") {
    a_type = CUDA_R_8I;
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Unsupported weight dtype %s.",
                                               weight_dtype));
  }
  if (std::is_same<T, phi::dtype::float16>::value) {
    bc_type = CUDA_R_16F;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    bc_type = CUDA_R_16BF;
  } else {
    PADDLE_THROW(common::errors::Unimplemented("Unsupported input dtype."));
  }
  cudaDataType_t Atype = a_type;
  cudaDataType_t Btype = bc_type;
  cudaDataType_t Ctype = bc_type;
  cudaDataType_t computeType = CUDA_R_32F;
  cudaDataType_t scaleType = CUDA_R_32F;
  cuinferGEMMCustomOption_t customOption;

  if (bias_data != nullptr) {
    customOption = CUINFER_BLAS_GEMM_CUSTOM_HALFBIAS;
  } else {
    customOption = CUINFER_BLAS_GEMM_CUSTOM_NONE;
  }

  int lda = k;
  int ldb = k;
  int ldc = n;
  float beta = 0.f;
  float alpha = 1.f;
  int batch_count = 1;

  cuinferQuantGEMMDeviceParam cust_device_param;
  cust_device_param.bias = bias_data;
  cust_device_param.scale = weight_scale_data;
  cust_device_param.workspace = nullptr;
  DenseTensor workspace_tensor;
  if (k % 64 != 0) {
    size_t workspace_size;
    cuinferStatus_t get_workspace_status =
        cuinferGetCustomGemmWorkspace(transa,
                                      transb,
                                      n,
                                      m,
                                      k,
                                      Atype,
                                      lda,
                                      lda,
                                      Btype,
                                      ldb,
                                      ldb,
                                      Ctype,
                                      ldc,
                                      ldc,
                                      batch_count,
                                      computeType,
                                      scaleType,
                                      &workspace_size);
    PADDLE_ENFORCE_EQ(
        get_workspace_status,
        CUINFER_STATUS_SUCCESS,
        common::errors::InvalidArgument(
            "IxInfer cuinferGetCustomGemmWorkspace calling failed with code %d",
            get_workspace_status));
    workspace_tensor.Resize({static_cast<int64_t>(workspace_size)});
    cust_device_param.workspace =
        static_cast<void*>(dev_ctx.template Alloc<uint8_t>(&workspace_tensor));
  }

  cuinferQuantGEMMHostParam cust_hots_param;
  cust_hots_param.size = sizeof(cuinferQuantGEMMHostParam);
  cust_hots_param.persistent = 0;
  cust_hots_param.groupSize = group_size;

  cuinferStatus_t status;
  // NN gemv int8 error
  status = cuinferCustomGemm(handle,
                             dev_ctx.stream(),
                             cuinfer_ptr_mode,
                             transa,
                             transb,
                             n,
                             m,
                             k,
                             &alpha,
                             weight_data,
                             Atype,
                             lda,
                             lda,
                             x_data,
                             Btype,
                             ldb,
                             ldb,
                             &beta,
                             out_data,
                             Ctype,
                             ldc,
                             ldc,
                             batch_count,
                             computeType,
                             scaleType,
                             &cust_hots_param,
                             &cust_device_param,
                             customOption);
  PADDLE_ENFORCE_EQ(
      status,
      CUINFER_STATUS_SUCCESS,
      common::errors::InvalidArgument(
          "IxInfer cuinferCustomGemm calling failed with code %d", status));
}
}  // namespace phi

PD_REGISTER_PLUGIN_KERNEL(weight_only_linear,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::WeightOnlyLinearKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
