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

#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include "kernels/funcs/mlu_funcs.h"
#include "paddle/phi/core/dense_tensor.h"

namespace custom_kernel {
using phi::CPUPlace;
using phi::DenseTensor;
const int64_t SAMPLE_MAX = 4;
template <typename T, typename Context>
void printInfo(const Context &dev_ctx, const DenseTensor &x, const std::string& name, bool frequency=false, bool shoud_sleep=false) {
    std::cout << "========================== START PRINT " << name << " ==========================" << std::endl;
    std::cout << "numel: "
              << x.numel()
              << std::endl;
    std::cout << "place: "
              << x.place()
              << std::endl;

    phi::DenseTensor tensor_tmp;
    phi::Copy(
        dev_ctx,
        x,
        CPUPlace(),
        true,
        &tensor_tmp);
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>) {
    T* data_p = static_cast<T*>(tensor_tmp.data());

    if(frequency){
      std::vector<int64_t> frequency(SAMPLE_MAX, 0);
      for (int i = 0; i < x.numel(); ++i) {
        if (data_p[i] >= 0 && data_p[i] < SAMPLE_MAX) {
          frequency[data_p[i]]++;
        } else {
          std::cout << "FOUND INVALID SAMPLE!" << std::endl;
          return ;
        }
      }
      std::cout << "frequency: " << std::endl;
      for (int i = 0; i < SAMPLE_MAX; ++i) {
        std::cout <<i << ": " << static_cast<float>(frequency[i]) / x.numel() << "\t";
      }
      std::cout << std::endl;
    }
    }

  std::cout<< "========================== END PRINT " << name << " ==========================" << std::endl << std::endl;
  if(shoud_sleep)
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
}

template <typename T, typename Context>
void MultinomialKernel(const Context &dev_ctx,
                       const phi::DenseTensor &x,
                       const phi::Scalar &num,
                       bool replacement,
                       phi::DenseTensor *out) {
  // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  dev_ctx.template Alloc<int64_t>(out);
  MLUCnnlTensorDesc desc_x(x);
  MLUCnnlTensorDesc desc_out(*out);

  int real_seed = static_cast<int>(dev_ctx.GetGenerator()->Random64());
  auto dev_id = static_cast<int64_t>(dev_ctx.GetPlace().GetDeviceId());
  auto generator_desc = GetMLURandomGenerator(dev_ctx, dev_id, real_seed);

  MLUCnnl::RandGenerateMultinomial(dev_ctx,
                                   generator_desc->get(),
                                   desc_x.get(),
                                   GetBasePtr(&x),
                                   replacement,
                                   false,
                                   GetBasePtr(&generator_desc->get_state()),
                                   desc_out.get(),
                                   GetBasePtr(out));
  std::cout << "End MultinomialKernel" << std::endl;
  // printInfo<T, Context>(dev_ctx, x, "x");
  // printInfo<int64_t, Context>(dev_ctx, *out, "out", true, false);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multinomial,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MultinomialKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
