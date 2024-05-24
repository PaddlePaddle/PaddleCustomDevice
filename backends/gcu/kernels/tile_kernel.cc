// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& repeat_times,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("tile");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor output_z =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    auto repeat_times_data = repeat_times.GetData();
    LAUNCH_TOPSATENOP(
        topsatenTile, dev_ctx, output_z, input_x, repeat_times_data);
    MaybeTransResult(dev_ctx, output_z, out);
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    auto repeat_times_data = repeat_times.GetData();
    attrs["repeat_times"] =
        std::vector<int>(repeat_times_data.begin(), repeat_times_data.end());

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "tile", dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tile,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TileKernel,
                          bool,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
