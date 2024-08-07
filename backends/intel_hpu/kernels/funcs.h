/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "runtime/runtime.h"

namespace custom_kernel {

/**
 * CPU -> INTEL_HPU
 * INTEL_HPU -> CPU
 * INTEL_HPU -> INTEL_HPU
//  */
// inline void TensorCopy(const phi::Context& dev_ctx,
//                        const phi::DenseTensor& src,
//                        bool blocking,
//                        phi::DenseTensor* dst,
//                        const phi::Place& dst_place = phi::CustomPlace()) {
// auto* src_ptr = src.data();
// const auto& src_place = src.place();
// auto dst_place_ = dst_place;
// if (dst_place_.GetType() != phi::AllocationType::CPU) {
//   dst_place_ = dev_ctx.GetPlace();
// }
// C_Device_st dst_device{dst_place_.GetDeviceId()};
// C_Device_st src_device{src_place.GetDeviceId()};

// VLOG(3) << "TensorCopy " << src.dims() << " from " << src_place << " to "
//         << dst_place_;

// dst->Resize(src.dims());
// void* dst_ptr = nullptr;
// if (dst_place_.GetType() != phi::AllocationType::CPU) {
//   dst_ptr = dev_ctx.Alloc(dst, src.dtype());
// } else {
//   dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
// }

// if (src_ptr == dst_ptr) {
//   VLOG(3) << "Skip copy the same data async from " << src_place << " to "
//           << src_place;
//   return;
// }
// VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

// C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());

// auto size = src.numel() * phi::SizeOf(src.dtype());

// if (src_place.GetType() == phi::AllocationType::CPU &&
//     dst_place_.GetType() == phi::AllocationType::CUSTOM) {
//   if (blocking) {
//     MemCpyH2D(&dst_device, dst_ptr, src_ptr, size);
//   } else {
//     AsyncMemCpyH2D(&dst_device, stream, dst_ptr, src_ptr, size);
//   }
// } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
//            dst_place_.GetType() == phi::AllocationType::CPU) {
//   if (blocking) {
//     MemCpyD2H(&src_device, dst_ptr, src_ptr, size);
//   } else {
//     AsyncMemCpyD2H(&src_device, stream, dst_ptr, src_ptr, size);
//   }
// } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
//            dst_place_.GetType() == phi::AllocationType::CUSTOM) {
//   if (src_place.GetDeviceType() == dst_place_.GetDeviceType()) {
//     if (src_place.GetDeviceId() == dst_place_.GetDeviceId()) {
//       if (blocking) {
//         MemCpyD2D(&src_device, dst_ptr, src_ptr, size);
//       } else {
//         AsyncMemCpyD2D(&src_device, stream, dst_ptr, src_ptr, size);
//       }
//     } else {
//       PADDLE_THROW(
//           phi::errors::Unimplemented("TensorCopy is not supported."));
//     }
//   } else {
//     PADDLE_THROW(phi::errors::Unimplemented("TensorCopy is not supported."));
//   }
// } else {
//   std::memcpy(dst_ptr, src_ptr, size);
// }
// }

}  // namespace custom_kernel
