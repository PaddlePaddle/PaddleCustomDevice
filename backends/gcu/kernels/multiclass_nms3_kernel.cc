// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
void MultiClassNMSKernel(const Context& dev_ctx,
                         const DenseTensor& bboxes,
                         const DenseTensor& scores,
                         const paddle::optional<DenseTensor>& rois_num,
                         float score_threshold,
                         int nms_top_k,
                         int keep_top_k,
                         float nms_threshold,
                         bool normalized,
                         float nms_eta,
                         int background_label,
                         DenseTensor* out,
                         DenseTensor* index,
                         DenseTensor* nms_rois_num) {
  PADDLE_GCU_KERNEL_TRACE("multiclass_nms3");
  if (LaunchAOTKernel()) {
    ContextPinnedGuard<Context> ctx_pinned_guard(dev_ctx);
    // Directly call the CPU implementation.
    VLOG(6) << "[CPU_KERNEL] Call CPU kernel for multiclass_nms3";

    // Copy bboxes to CPU
    DenseTensor bboxes_cpu;
    DenseTensor bboxes_gcu;
    if (bboxes.dtype() == DataType::FLOAT16) {
      custom_kernel::Cast(dev_ctx, bboxes, DataType::FLOAT32, &bboxes_gcu);
    } else {
      bboxes_gcu = bboxes;
    }
    TensorCopy(dev_ctx, bboxes_gcu, false, &bboxes_cpu, CPUPlace());

    // Copy scores to CPU
    DenseTensor scores_cpu;
    DenseTensor scores_gcu;
    if (scores.dtype() == DataType::FLOAT16) {
      custom_kernel::Cast(dev_ctx, scores, DataType::FLOAT32, &scores_gcu);
    } else {
      scores_gcu = scores;
    }
    TensorCopy(dev_ctx, scores_gcu, false, &scores_cpu, CPUPlace());

    // Copy rois_num to CPU if need
    paddle::optional<DenseTensor> rois_num_cpu =
        paddle::optional<DenseTensor>();
    if (rois_num) {
      DenseTensor rois_num_tensor = rois_num.get();
      DenseTensor rois_num_tensor_cpu;
      DenseTensor rois_num_tensor_gcu;
      if (rois_num_tensor.dtype() == DataType::FLOAT16) {
        custom_kernel::Cast(
            dev_ctx, rois_num_tensor, DataType::FLOAT32, &rois_num_tensor_gcu);
      } else {
        rois_num_tensor_gcu = rois_num_tensor;
      }
      TensorCopy(dev_ctx,
                 rois_num_tensor_gcu,
                 false,
                 &rois_num_tensor_cpu,
                 CPUPlace());
      rois_num_cpu = paddle::make_optional<DenseTensor>(rois_num_tensor_cpu);
    }

    // Wait for data ready
    dev_ctx.Wait();

    DenseTensor out_cpu = *out;
    if (out->dtype() == DataType::FLOAT16) {
      DenseTensorMeta cpu_meta(DataType::FLOAT32, out->dims());
      out_cpu.set_meta(cpu_meta);
    }
    DenseTensor index_cpu = *index;
    DenseTensor nms_rois_num_cpu = *nms_rois_num;

    // Call the CPU implementation
    CPUContext dev_ctx_cpu;
    dev_ctx_cpu.SetAllocator(&(dev_ctx.GetHostAllocator()));
    dev_ctx_cpu.SetHostAllocator(&(dev_ctx.GetHostAllocator()));

    phi::MultiClassNMSKernel<float, CPUContext>(dev_ctx_cpu,
                                                bboxes_cpu,
                                                scores_cpu,
                                                rois_num_cpu,
                                                score_threshold,
                                                nms_top_k,
                                                keep_top_k,
                                                nms_threshold,
                                                normalized,
                                                nms_eta,
                                                background_label,
                                                &out_cpu,
                                                &index_cpu,
                                                &nms_rois_num_cpu);

    dev_ctx.Wait();

    // convert result
    DenseTensor out_gcu;
    TensorCopy(dev_ctx, out_cpu, false, &out_gcu);
    if (out->dtype() == DataType::FLOAT16) {
      custom_kernel::Cast(dev_ctx, out_gcu, DataType::FLOAT16, out);
    } else {
      *out = out_gcu;
    }
    TensorCopy(dev_ctx, index_cpu, false, index);
    TensorCopy(dev_ctx, nms_rois_num_cpu, false, nms_rois_num);

    dev_ctx.Wait();

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multiclass_nms3,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MultiClassNMSKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
