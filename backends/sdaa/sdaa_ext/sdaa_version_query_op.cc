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

#include <vector>

#include "paddle/extension.h"
#include "sdaa_runtime.h"  //NOLINT
#include "tecoblas.h"      //NOLINT
#include "tecodnn.h"       //NOLINT
#include "tools/version/query.h"

#define CHECK_CPU_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a cpu Tensor.")

std::vector<std::vector<int64_t>> VersionInferShape(
    const std::vector<int64_t>& x_shape) {
  return {{-1}};
}

std::vector<paddle::DataType> VersionInferDtype(
    const paddle::DataType& x_dtype) {
  return {paddle::DataType::UINT8};
}

std::vector<paddle::Tensor> CustomGetPaddleCompilationVersion() {
  auto paddlepaddle_version = GetPaddlePaddleVersion().version;
  auto len = paddlepaddle_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = paddlepaddle_version[i];
  }
  return {out};
}

std::vector<paddle::Tensor> CustomGetPaddleCommitCompilationVersion() {
  auto paddlecommit_version = GetPaddlePaddleCommit().version;
  auto len = paddlecommit_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = paddlecommit_version[i];
  }
  return {out};
}

std::vector<paddle::Tensor> CustomGetSdaaDriverVersion() {
  auto sdaa_driver_version = GetSdaaDriverVersion().version;
  auto len = sdaa_driver_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = sdaa_driver_version[i];
  }
  return {out};
}

std::vector<paddle::Tensor> CustomGetSdaaRuntimeVersion() {
  auto sdaa_runtime_version = GetSdaaRuntimeVersion().version;
  auto len = sdaa_runtime_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = sdaa_runtime_version[i];
  }
  return {out};
}

std::vector<paddle::Tensor> CustomGetTecoDNNVersion() {
  auto teco_dnn_version = GetTecoDNNVersion().version;
  auto len = teco_dnn_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = teco_dnn_version[i];
  }
  return {out};
}

std::vector<paddle::Tensor> CustomGetTecoBLASVersion() {
  auto teco_blas_version = GetTecoBLASVersion().version;
  auto len = teco_blas_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = teco_blas_version[i];
  }
  return {out};
}

std::vector<paddle::Tensor> CustomGetTecoCustomVersion() {
  auto teco_custom_version = GetTecoCustomVersion().version;
  auto len = teco_custom_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = teco_custom_version[i];
  }
  return {out};
}

std::vector<paddle::Tensor> CustomGetTCCLVersion() {
  auto tccl_version = GetTCCLVersion().version;
  auto len = tccl_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = tccl_version[i];
  }
  return {out};
}

std::vector<paddle::Tensor> CustomGetSDptiVersion() {
  auto sdpti_version = GetSDptiVersion().version;
  auto len = sdpti_version.size();
  auto out = paddle::zeros({len}, paddle::DataType::UINT8);
  auto* out_data = out.data<uint8_t>();
  for (int64_t i = 0; i < len; i++) {
    out_data[i] = sdpti_version[i];
  }
  return {out};
}

PD_BUILD_OP(custom_sdaa_driver_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetSdaaDriverVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));

PD_BUILD_OP(custom_paddle_compilation_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetPaddleCompilationVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));

PD_BUILD_OP(custom_paddle_commit_compilation_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetPaddleCommitCompilationVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));

PD_BUILD_OP(custom_sdaa_runtime_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetSdaaRuntimeVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));

PD_BUILD_OP(custom_teco_dnn_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetTecoDNNVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));

PD_BUILD_OP(custom_teco_blas_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetTecoBLASVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));

PD_BUILD_OP(custom_teco_custom_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetTecoCustomVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));

PD_BUILD_OP(custom_tccl_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetTCCLVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));

PD_BUILD_OP(custom_sdpti_version)
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomGetSDptiVersion))
    .SetInferShapeFn(PD_INFER_SHAPE(VersionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(VersionInferDtype));
