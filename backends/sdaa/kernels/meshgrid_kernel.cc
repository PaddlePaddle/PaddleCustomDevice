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

#include <iostream>
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void MeshgridKernel(const Context& dev_ctx,
                    const std::vector<const phi::DenseTensor*>& inputs,
                    std::vector<phi::DenseTensor*> outputs) {
  VLOG(4) << "Call SDAA MeshgridKernel";

  int rank = inputs.size();
  std::vector<void*> input_ptr;
  std::vector<void*> output_ptr;
  std::vector<tecodnnTensorDescriptor_t> inputDesc;
  std::vector<tecodnnTensorDescriptor_t> outputDesc;

  for (int i = 0; i < rank; i++) {
    dev_ctx.template Alloc<T>(outputs[i]);
    std::vector<int> every_input_dims = phi::vectorize<int>(inputs[i]->dims());
    std::vector<int> every_output_dims =
        phi::vectorize<int>(outputs[i]->dims());
    void* every_input_ptr = const_cast<void*>(inputs[i]->data());
    void* every_output_ptr = const_cast<void*>(outputs[i]->data());

    input_ptr.push_back(every_input_ptr);
    output_ptr.push_back(every_output_ptr);
    tecodnnTensorDescriptor_t every_input_Desc = sdaa_ops::GetTecodnnTensorDesc(
        every_input_dims, inputs[i]->dtype(), TensorFormat::Undefined);
    tecodnnTensorDescriptor_t every_output_Desc =
        sdaa_ops::GetTecodnnTensorDesc(
            every_output_dims, outputs[i]->dtype(), TensorFormat::Undefined);
    inputDesc.push_back(every_input_Desc);
    outputDesc.push_back(every_output_Desc);
  }

  int inputWorkspaceSize = input_ptr.size() * sizeof(void*);
  int outputWorkspaceSize = output_ptr.size() * sizeof(void*);
  int64_t hostInputSize = inputWorkspaceSize;
  int64_t hostOutputSize = outputWorkspaceSize;
  std::vector<int8_t> host_input(hostInputSize);
  std::vector<int8_t> host_output(hostOutputSize);
  memcpy(host_input.data(), input_ptr.data(), inputWorkspaceSize);
  memcpy(host_output.data(), output_ptr.data(), outputWorkspaceSize);

  phi::DenseTensor input_tmp, output_tmp;
  input_tmp.Resize(phi::make_ddim({hostInputSize}));
  output_tmp.Resize(phi::make_ddim({hostOutputSize}));
  dev_ctx.Alloc(&input_tmp, phi::DataType::INT8);
  dev_ctx.Alloc(&output_tmp, phi::DataType::INT8);
  AsyncMemCpyH2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 input_tmp.data(),
                 host_input.data(),
                 hostInputSize);
  AsyncMemCpyH2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 output_tmp.data(),
                 host_output.data(),
                 hostOutputSize);

  void** input_ptr_void = reinterpret_cast<void**>(input_tmp.data<int8_t>());
  void** output_ptr_void = reinterpret_cast<void**>(output_tmp.data<int8_t>());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  TECODNN_CHECK(tecodnnMeshgrid(tecodnnHandle,
                                rank,
                                inputDesc.data(),
                                input_ptr_void,
                                outputDesc.data(),
                                output_ptr_void));
  for (int i = 0; i < rank; i++) {
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(inputDesc[i]));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(outputDesc[i]));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(meshgrid,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MeshgridKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
