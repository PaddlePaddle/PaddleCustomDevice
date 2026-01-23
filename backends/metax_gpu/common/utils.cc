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

#include "common/utils.h"
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

#include "glog/logging.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/custom/custom_context.h"
namespace phi {
namespace {
C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void* dst,
                        const void* src,
                        size_t size) {
  if (size == 0) {
    return C_SUCCESS;
  }

  if (dst == NULL || src == NULL) {
    return C_ERROR;
  }
  cudaError_t cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  cudaErr = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void* dst,
                        const void* src,
                        size_t size) {
  if (size == 0) {
    return C_SUCCESS;
  }

  if (dst == NULL || src == NULL) {
    return C_ERROR;
  }

  cudaError_t cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  cudaErr = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void* dst,
                        const void* src,
                        size_t size) {
  if (size == 0) {
    VLOG(2) << "cudamemcpy successful: " << dst << " " << src << " "
            << size;  // NOLINT
    return C_SUCCESS;
  }

  if (dst == NULL || src == NULL) {
    return C_ERROR;
  }

  cudaError_t cudaErr = cudaSetDevice(device->id);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }

  cudaErr = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice);
  if (cudaErr != cudaSuccess) {
    return C_ERROR;
  }
  VLOG(2) << "cudamemcpy successful: " << dst << " " << src << " "
          << size;  // NOLINT
  return C_SUCCESS;
}

template <typename Context>
inline void TensorCopy(const Context& dev_ctx,
                       const phi::DenseTensor& src,
                       bool blocking,
                       phi::DenseTensor* dst,
                       const phi::Place& dst_place = phi::CustomPlace()) {
  auto* src_ptr = src.data();
  const auto& src_place = src.place();
  if (src_ptr == nullptr) {
    return;
  }
  auto dst_place_ = dst_place;
  if (dst_place_.GetType() != phi::AllocationType::CPU) {
    dst_place_ = dev_ctx.GetPlace();
  }

  if (&src == dst) {
    if (src_place == dst_place_) {
      VLOG(6) << "Skip copy the same data(" << src_ptr << ") from " << src_place
              << " to " << dst_place_;
    } else {
      VLOG(6) << "Src and dst are the same Tensor, in-place copy data("
              << src_ptr << ") from " << src_place << " to " << dst_place_;
      const phi::DenseTensor src_copy = src;
      TensorCopy(dev_ctx, src_copy, blocking, dst, dst_place_);
    }
    return;
  }

  auto dst_dims = dst->dims();
  dst->Resize(src.dims());
  void* dst_ptr = nullptr;
  if (dst_place_.GetType() != phi::AllocationType::CPU) {
    dst_ptr = dev_ctx.Alloc(dst, src.dtype());
  } else {
    dst_ptr = dev_ctx.HostAlloc(dst, src.dtype());
  }

  PADDLE_ENFORCE_EQ(
      dst->place(),
      dst_place_,
      phi::errors::Unavailable(
          "The Dst Tensor's place and dst_place do not match, Tensor's place "
          "place is %s, dst_place is %s.",
          dst->place(),
          dst_place_));

  if (src_ptr == dst_ptr && src_place == dst_place_) {
    if ((dst_dims == src.dims()) || (src_place == phi::CPUPlace())) {
      VLOG(3) << "Skip copy the same data async from " << src_ptr << " in "
              << src_place << " to " << dst_ptr << " in " << dst_place_;
      return;
    } else {
      // scatter memory
      phi::DenseTensor tmp_dst;
      tmp_dst.set_meta(dst->meta());
      tmp_dst.Resize(dst_dims);
      dst_ptr = dev_ctx.Alloc(&tmp_dst, tmp_dst.dtype());
      *dst = tmp_dst;
    }
  }
  VLOG(4) << "src:" << src_ptr << " place: " << src_place
          << " type:" << static_cast<int>(src_place.GetType())
          << ", dst:" << dst_ptr << " place: " << dst_place_
          << " type:" << static_cast<int>(dst_place_.GetType());

  C_Stream stream = reinterpret_cast<C_Stream>(dev_ctx.stream());

  auto size =
      (src.dims().size() != 0 ? src.numel() : 1) * phi::SizeOf(src.dtype());
  if (UNLIKELY(size) == 0) {
    return;
  }

  if (src_place.GetType() == phi::AllocationType::CPU &&
      dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    VLOG(6) << "TensorCopy from cpu to cus";
    C_Device_st device;
    device.id = dst_place_.GetDeviceId();
    AsyncMemCpyH2D(&device, stream, dst_ptr, src_ptr, size);
    if (blocking) {
      dev_ctx.Wait();
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    VLOG(6) << "TensorCopy from cus to cpu";
    C_Device_st device;
    device.id = src_place.GetDeviceId();
    AsyncMemCpyD2H(&device, stream, dst_ptr, src_ptr, size);
    if (blocking) {
      dev_ctx.Wait();
    }
  } else if (src_place.GetType() == phi::AllocationType::CUSTOM &&
             dst_place_.GetType() == phi::AllocationType::CUSTOM) {
    VLOG(6) << "TensorCopy from cus to cus";
    if (src_place.GetDeviceType() == dst_place_.GetDeviceType()) {
      if (src_place.GetDeviceId() == dst_place_.GetDeviceId()) {
        C_Device_st device;
        device.id = src_place.GetDeviceId();
        AsyncMemCpyD2D(&device, stream, dst_ptr, src_ptr, size);
        if (blocking) {
          dev_ctx.Wait();
        }
      } else {
        PADDLE_THROW(
            phi::errors::Unimplemented("TensorCopy is not supported."));
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("TensorCopy is not supported."));
    }
  } else if (src_place.GetType() == phi::AllocationType::CPU &&
             dst_place_.GetType() == phi::AllocationType::CPU) {
    VLOG(6) << "TensorCopy from cpu to cpu";
    std::memcpy(dst_ptr, src_ptr, size);
  }
}

template <typename T = float>
std::ostream& PrintTensor(std::ostream& os, const phi::DenseTensor& tensor) {
  phi::DenseTensor cpu_tensor;
  if (tensor.place().GetType() != phi::AllocationType::CPU) {
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        phi::DeviceContextPool::Instance().Get(tensor.place()));
    TensorCopy(*dev_ctx, tensor, true, &cpu_tensor, phi::CPUPlace());
  } else {
    cpu_tensor = tensor;
  }
  os << "DenseTensor<";
  if (tensor.initialized()) {
    os << phi::DataTypeToString(tensor.dtype()) << ", ";
    os << tensor.place() << ", ";
    os << "Shape(" << tensor.dims() << "), ";
    os << "Strides(" << tensor.strides() << "), ";
    os << "layout:" << tensor.layout() << ", ";
    os << "data: [";

    auto ptr = cpu_tensor.data<T>();
    auto element_num = cpu_tensor.numel();
    // Note: int8_t && uint8_t is typedef of char, ostream unable to print
    // properly
    if (typeid(int8_t) == typeid(T) || typeid(uint8_t) == typeid(T)) {
      if (element_num > 0) {
        os << signed(ptr[0]);
        for (int j = 1; j < element_num; ++j) {
          os << " " << signed(ptr[j]);
        }
      }
    } else {
      if (element_num > 0) {
        os << ptr[0];
        for (int j = 1; j < element_num; ++j) {
          os << " " << ptr[j];
        }
      }
    }
    os << "]";
  } else {
    os << "NOT_INITED";
  }
  os << ">";
  return os;
}
}  // namespace

#define FOR_EACH_DATA_TYPE_TO_PRINT(_)      \
  _(bool, phi::DataType::BOOL)              \
  _(int8_t, phi::DataType::INT8)            \
  _(uint8_t, phi::DataType::UINT8)          \
  _(int16_t, phi::DataType::INT16)          \
  _(uint16_t, phi::DataType::UINT16)        \
  _(int32_t, phi::DataType::INT32)          \
  _(uint32_t, phi::DataType::UINT32)        \
  _(int64_t, phi::DataType::INT64)          \
  _(uint64_t, phi::DataType::UINT64)        \
  _(phi::bfloat16, phi::DataType::BFLOAT16) \
  _(phi::float16, phi::DataType::FLOAT16)   \
  _(float, phi::DataType::FLOAT32)          \
  _(double, phi::DataType::FLOAT64)

#define CALL_PRINT_TENSOR(cpp_type, data_type) \
  case data_type:                              \
    PrintTensor<cpp_type>(os, t);              \
    break;

std::ostream& operator<<(std::ostream& os, const phi::DenseTensor& t) {
  switch (t.dtype()) {
    FOR_EACH_DATA_TYPE_TO_PRINT(CALL_PRINT_TENSOR)
    default:
      VLOG(1) << "PrintTensor unrecognized data type:" << t.dtype();
  }
  return os;
}
#undef FOR_EACH_DATA_TYPE_TO_PRINT
#undef CALL_PRINT_TENSOR
}  // namespace phi

// lock_mcruntime.c

__attribute__((constructor)) void lock_mcruntime() {
  void* handle =
      dlopen("/opt/maca/lib/libmcruntime.so", RTLD_LAZY | RTLD_NODELETE);
  if (!handle) {
    fprintf(stderr, "Failed to lock libmcruntime.so: %s\n", dlerror());
  } else {
  }
}
