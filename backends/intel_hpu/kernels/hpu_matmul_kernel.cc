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
#include "funcs.h"
#include "hpu_operator.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "synapse_common_types.h"

namespace custom_kernel {

class BatchGEMM : public HpuOperator {
 public:
  BatchGEMM(synDataType dtype) : HpuOperator("batch_gemm"), dtype_(dtype) {}

  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               bool trans_x,
               bool trans_y) {
    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), dtype_, ins[0], true, "x"),
        createTensor(ins[1].size(), dtype_, ins[1], true, "y")};
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), dtype_, outs[0], true, "output")};
    synGEMMParams params{trans_x, trans_y};
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "BATCH_GEMM",
                                     nullptr,
                                     nullptr);
    LOG_IF(ERROR, status != synSuccess)
        << "[RUNTIME] synNodeCreate() failed = " << status;
  }

 protected:
  synDataType dtype_;
};

class GEMM : public HpuOperator {
 public:
  GEMM(synDataType dtype) : HpuOperator("gemm"), dtype_(dtype) {}

  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               bool trans_x,
               bool trans_y) {
    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), dtype_, ins[0], true, "x"),
        createTensor(ins[1].size(), dtype_, ins[1], true, "y")};
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), dtype_, outs[0], true, "output")};
    synGEMMParams params{trans_x, trans_y};
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "GEMM",
                                     nullptr,
                                     nullptr);
    LOG_IF(ERROR, status != synSuccess)
        << "[RUNTIME] synNodeCreate() failed = " << status;
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {
  auto x_rank = x.dims().size();
  auto y_rank = y.dims().size();
  VLOG(9) << " x rank = " << x_rank << ", y rank = " << y_rank;
  PD_CHECK((x_rank == y_rank) || (y_rank != 2 && x_rank != y_rank),
           "matmul rank not support");

  synDataType dtype = syn_type_na;
  if (std::is_same<T, phi::dtype::float16>::value) {
    dtype = syn_type_fp16;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    dtype = syn_type_bf16;
  } else if (std::is_same<T, phi::dtype::float8_e4m3fn>::value) {
    dtype = syn_type_fp8_143;
  }

  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize<int64_t>(y.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  std::map<std::string, uint64_t> tensors;
  tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
  tensors["y"] = reinterpret_cast<uint64_t>(y.data<T>());
  tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());

  if (y_rank == 2) {
    // gemm
    GEMM gemm(dtype);
    gemm.AddNode({x_dims, y_dims}, {outputs_dim}, transpose_x, transpose_y);
    gemm.Compile();
    gemm.Execute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  } else {
    // batch gemm
    BatchGEMM batch_gemm(dtype);
    batch_gemm.AddNode(
        {x_dims, y_dims}, {outputs_dim}, transpose_x, transpose_y);
    batch_gemm.Compile();
    batch_gemm.Execute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(matmul,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MatmulKernel,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          phi::dtype::float8_e4m3fn) {}
