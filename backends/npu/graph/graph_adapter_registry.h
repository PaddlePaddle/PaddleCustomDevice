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

#pragma once

#define USE_OP_ADAPTER(ir)                                              \
  extern int __op_adapter_registrar_##ir##__touch__();                  \
  static __attribute__((unused)) int __use_op_adapter_##ir##__touch__ = \
      __op_adapter_registrar_##ir##__touch__()

USE_OP_ADAPTER(feed);
USE_OP_ADAPTER(fetch_v2);
USE_OP_ADAPTER(relu);
USE_OP_ADAPTER(relu_grad);
USE_OP_ADAPTER(elementwise_add);
USE_OP_ADAPTER(elementwise_add_grad);
USE_OP_ADAPTER(reduce_mean);
USE_OP_ADAPTER(reduce_mean_grad);
USE_OP_ADAPTER(conv2d);
USE_OP_ADAPTER(conv2d_grad);
USE_OP_ADAPTER(fill_constant);
USE_OP_ADAPTER(flatten_contiguous_range);
USE_OP_ADAPTER(flatten_contiguous_range_grad);
USE_OP_ADAPTER(mul);
USE_OP_ADAPTER(mul_grad);
USE_OP_ADAPTER(pool2d);
USE_OP_ADAPTER(pool2d_grad);
USE_OP_ADAPTER(uniform_random);
USE_OP_ADAPTER(gaussian_random);
USE_OP_ADAPTER(matmul_v2);
USE_OP_ADAPTER(matmul_v2_grad);
USE_OP_ADAPTER(top_k_v2);
USE_OP_ADAPTER(accuracy);
USE_OP_ADAPTER(softmax_with_cross_entropy);
USE_OP_ADAPTER(softmax_with_cross_entropy_grad);
USE_OP_ADAPTER(sgd);
USE_OP_ADAPTER(abs);
USE_OP_ADAPTER(adam);
USE_OP_ADAPTER(batch_norm);
USE_OP_ADAPTER(batch_norm_grad);
USE_OP_ADAPTER(mean);
USE_OP_ADAPTER(mean_grad);
