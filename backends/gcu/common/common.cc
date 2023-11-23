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

#include "common/common.h"

#include <dtu/driver/device_manager.h>
#include <gcu/umd/device_ids.h>

#include <algorithm>
#include <string>
#include <vector>

#include "backend/utils/utils.h"
#include "dtu/hlir/library.h"
#include "dtu/op_define/type.h"
#include "kernels/funcs/gcu_funcs.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/ddim.h"

namespace custom_kernel {
namespace {
enum class ChipType {
  LEO,
  PAVO,
  PAVO_1C,
  DORADO,
  DORADO_2C,
  DORADO_3PG,
  LIBRA,
  SCORPIO,
  UNKNOW,
};

const static std::string KDev = "dtu";  // NOLINT
}  // namespace

static inline std::vector<int64_t> contiguous_strides(
    const std::vector<int64_t>& sizes) {
  const int64_t ndims = static_cast<int64_t>(sizes.size());
  std::vector<int64_t> strides(ndims, 1);
  for (auto i = ndims - 2; i >= 0; --i) {
    // strides can't be 0 even if sizes are 0.
    strides[i] = strides[i + 1] * std::max(sizes[i + 1], int64_t{1});
  }
  return strides;
}

static inline std::vector<int64_t> contiguous_strides(
    const std::vector<int64_t>& sizes, const std::vector<int32_t>& layouts) {
  VLOG(6) << "contiguous_strides layouts: " << backend::VectorToString(layouts)
          << " sizes: " << backend::VectorToString(sizes);
  auto rank = sizes.size();
  PADDLE_ENFORCE_EQ(
      rank,
      layouts.size(),
      phi::errors::InvalidArgument("%d vs %d", rank, layouts.size()));
  auto cstrides = contiguous_strides(sizes);
  auto strides = cstrides;
  for (size_t i = 0; i < rank; i++) {
    strides[layouts[i]] = cstrides[i];
  }
  return strides;
}

static inline std::vector<uint8_t> contiguous_layouts(const size_t ndims) {
  PADDLE_ENFORCE_LT(ndims,
                    UINT8_MAX,
                    phi::errors::InvalidArgument(" with ndims = %d", ndims));
  std::vector<uint8_t> contiguous_layout(ndims, 0);
  std::iota(contiguous_layout.begin(), contiguous_layout.end(), 0);
  return contiguous_layout;
}

static inline std::vector<int64_t> contiguous_layouts_ex(const size_t ndims) {
  std::vector<int64_t> contiguous_layout(ndims, 0);
  std::iota(contiguous_layout.begin(), contiguous_layout.end(), 0);
  return contiguous_layout;
}

void* GetOpFuncPtr(const std::string& name,
                   const hlir::DispatchParam& params,
                   bool include_output) {
  auto tensors = params.inputs;
  if (include_output) {
    tensors.insert(tensors.end(), params.outputs.begin(), params.outputs.end());
  }
  auto func_ptr =
      hlir::getLibrary(name.c_str(),
                       KDev.c_str(),
                       tensors,
                       const_cast<hlir::Metadata*>(&params.metadata));
  return func_ptr;
}

void* GetOpFuncPtr(const std::string& name,
                   const hlir::DispatchParam& params,
                   const std::vector<hlir::Tensor*>& tensors) {
  auto func_ptr =
      hlir::getLibrary(name.c_str(),
                       KDev.c_str(),
                       tensors,
                       const_cast<hlir::Metadata*>(&params.metadata));
  return func_ptr;
}

GcuMemory* GetGcuMemory(const phi::DenseTensor& tensor, bool check_place) {
  if (check_place) {
    PADDLE_ENFORCE_EQ(
        tensor.place().GetDeviceType(),
        "gcu",
        phi::errors::InvalidArgument("not gcu tensor, current tensor place: %s",
                                     tensor.place().GetDeviceType().c_str()));
  }

  if (tensor.initialized()) {
    return static_cast<GcuMemory*>(const_cast<void*>(tensor.data()));
  } else {
    return nullptr;
  }
}

int GetGCUDataType(const phi::DataType& dtype) {
  switch (dtype) {
    case phi::DataType::UINT8:
      return hlir::U8;
    case phi::DataType::INT8:
      return hlir::S8;
    case phi::DataType::INT16:
      return hlir::S16;
    case phi::DataType::UINT16:
      return hlir::U16;
    case phi::DataType::INT32:
      return hlir::S32;
    case phi::DataType::UINT32:
      return hlir::U32;
    case phi::DataType::INT64:
      return hlir::S64;
    case phi::DataType::UINT64:
      return hlir::U64;
    case phi::DataType::FLOAT16:
      return hlir::F16;
    case phi::DataType::FLOAT32:
      return hlir::F32;
    case phi::DataType::FLOAT64:
      return hlir::F64;
    case phi::DataType::COMPLEX64:
      return hlir::C64;
    case phi::DataType::COMPLEX128:
      return hlir::C128;
    case phi::DataType::BOOL:
      return hlir::PRED;
    case phi::DataType::BFLOAT16:
      return hlir::BF16;
    default: {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("Invalid scalar type %s",
                                       DataTypeToString(dtype).c_str()));
      return hlir::PRIMITIVE_TYPE_INVALID;
    }
  }
}

std::vector<int32_t> LayoutToVector(phi::DataLayout layout) {
  if (layout == phi::DataLayout::NHWC) {
    return {0, 2, 3, 1};
  } else if (layout == phi::DataLayout::NCHW) {
    return {0, 1, 2, 3};
  } else {
    phi::errors::Fatal("unsupport layout %s",
                       phi::DataLayoutToString(layout).c_str());
    return {};
  }
}

phi::DataLayout VectorToLayout(const std::vector<int32_t>& layout) {
  static std::vector<int32_t> nhwc = {0, 2, 3, 1};
  static std::vector<int32_t> nchw = {0, 1, 2, 3};
  if (layout == nhwc) {
    return phi::DataLayout::NHWC;
  } else if (layout == nchw) {
    return phi::DataLayout::NCHW;
  } else {
    phi::errors::Fatal("unsupport layout %s",
                       backend::VectorToString(layout).c_str());
    return phi::DataLayout::UNDEFINED;
  }
}

void LayoutConvertDims(const std::vector<int64_t>& dims,
                       const std::vector<int32_t>& src_layout,
                       const std::vector<int32_t>& dst_layout,
                       std::vector<int64_t>& out_permute_dims,    // NOLINT
                       std::vector<int64_t>& out_convert_dims) {  // NOLINT
  PADDLE_ENFORCE_EQ(dims.size(),
                    src_layout.size(),
                    phi::errors::InvalidArgument(
                        "dims size %d not equal to src layout size %d",
                        dims.size(),
                        src_layout.size()));

  out_permute_dims = LayoutAffine(src_layout, dst_layout);

  PADDLE_ENFORCE_EQ(dims.size(),
                    out_permute_dims.size(),
                    phi::errors::InvalidArgument(
                        "dims size %d not equal to permute dims size %d",
                        dims.size(),
                        out_permute_dims.size()));

  out_convert_dims.resize(dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    out_convert_dims[i] = dims.at(out_permute_dims[i]);
  }
}

std::vector<int64_t> LayoutAffine(const std::vector<int32_t>& src_layout,
                                  const std::vector<int32_t>& dst_layout) {
  PADDLE_ENFORCE_EQ(src_layout.size(),
                    dst_layout.size(),
                    phi::errors::InvalidArgument(
                        "src layout size %d not equal to dst layout size %d",
                        src_layout.size(),
                        dst_layout.size()));

  std::vector<int64_t> permute_dims;
  auto ndims = src_layout.size();
  permute_dims.resize(ndims);

  auto idx_find = [](const std::vector<int32_t>& layout, int32_t data) {
    for (size_t j = 0; j < layout.size(); j++) {
      if (layout[j] == data) {
        return j;
      }
    }
    phi::errors::Fatal(
        "can not find %d in %s", data, backend::VectorToString(layout).c_str());
    return 0UL;  // Just for return-type warning, will not reach here.
  };

  for (size_t i = 0; i < ndims; i++) {
    permute_dims[i] = idx_find(src_layout, dst_layout[i]);
  }

  return permute_dims;
}

hlir::Tensor* GetHlirTensor(const phi::DenseTensor& tensor) {
  PADDLE_ENFORCE_EQ(
      tensor.place().GetDeviceType(),
      "gcu",
      phi::errors::InvalidArgument("not gcu tensor, current tensor place: %s",
                                   tensor.place().GetDeviceType().c_str()));
  auto gcu_memory = GetGcuMemory(tensor, false);
  void* data_ptr = nullptr;
  std::vector<int64_t> dims = {};
  if (gcu_memory != nullptr) {
    data_ptr = gcu_memory->mem_ptr;

    // size_t ndims = 16;
    // std::vector<int64_t> tmp_dims(ndims, 0);
    // RT_CHECK(topsMemoryGetDims(data_ptr, (int64_t*)tmp_dims.data(), &ndims));
    // dims = std::vector<int64_t>(tmp_dims.begin(), tmp_dims.begin() + ndims);
    dims = gcu_memory->dims;
  }

  std::vector<int64_t> tensor_dims = phi::vectorize(tensor.dims());
  if (tensor_dims.empty()) {
    tensor_dims = {1};
  }

  PADDLE_ENFORCE_EQ(
      tensor_dims,
      dims,
      phi::errors::NotFound("tensor dims is %s, gcu tensor dims is %s, tensor "
                            "ptr: %lu, scatter mem ptr: %lu ",
                            phi::make_ddim(tensor_dims).to_str().c_str(),
                            phi::make_ddim(dims).to_str().c_str(),
                            gcu_memory,
                            data_ptr));

  auto strides = contiguous_strides(dims);
  auto layouts = contiguous_layouts_ex(dims.size());
  dims = dims.empty() ? std::vector<int64_t>{1} : dims;
  strides = strides.empty() ? std::vector<int64_t>{1} : strides;
  layouts = layouts.empty() ? std::vector<int64_t>{0} : layouts;
  return (new hlir::Tensor{data_ptr,
                           GetGCUDataType(tensor.dtype()),
                           static_cast<int64_t>(tensor.capacity()),
                           dims,
                           strides,
                           layouts});
}

hlir::Tensor* GetHlirTensorV2(const phi::DenseTensor& tensor,
                              const phi::DDim src_dims) {
  PADDLE_ENFORCE_EQ(
      tensor.place().GetDeviceType(),
      "gcu",
      phi::errors::InvalidArgument("not gcu tensor, current tensor place: %s",
                                   tensor.place().GetDeviceType().c_str()));
  auto gcu_memory = GetGcuMemory(tensor, false);
  void* data_ptr = nullptr;
  std::vector<int64_t> dims = {};
  if (gcu_memory != nullptr) {
    data_ptr = gcu_memory->mem_ptr;

    // size_t ndims = 16;
    // std::vector<int64_t> tmp_dims(ndims, 0);
    // RT_CHECK(topsMemoryGetDims(data_ptr, (int64_t*)tmp_dims.data(), &ndims));
    // dims = std::vector<int64_t>(tmp_dims.begin(), tmp_dims.begin() + ndims);
    dims = gcu_memory->dims;
  }

  std::vector<int64_t> tensor_dims = phi::vectorize(tensor.dims());
  if (tensor_dims.empty()) {
    tensor_dims = {1};
  }

  PADDLE_ENFORCE_EQ(
      tensor_dims,
      dims,
      phi::errors::NotFound("tensor dims is %s, gcu tensor dims is %s",
                            phi::make_ddim(tensor_dims).to_str().c_str(),
                            phi::make_ddim(dims).to_str().c_str()));

  auto strides = contiguous_strides(dims, LayoutToVector(tensor.layout()));
  dims = phi::vectorize(src_dims);
  auto layouts = contiguous_layouts_ex(dims.size());
  dims = dims.empty() ? std::vector<int64_t>{1} : dims;
  strides = strides.empty() ? std::vector<int64_t>{1} : strides;
  layouts = layouts.empty() ? std::vector<int64_t>{0} : layouts;
  return (new hlir::Tensor{data_ptr,
                           GetGCUDataType(tensor.dtype()),
                           static_cast<int64_t>(tensor.capacity()),
                           dims,
                           strides,
                           layouts});
}

bool GcuOpStreamSync(const phi::DeviceContext& dev_ctx) {
  static bool use_sync = true;  // need get from env
  if (use_sync) {
    dev_ctx.Wait();
    return true;
  }

  return false;
}

void BuildDispatchParam(const std::vector<LoDTensor*>& inputs,
                        const std::vector<LoDTensor*>& outputs,
                        hlir::DispatchParam& params) {  // NOLINT
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor = inputs[i];
    VLOG(6) << "start get input hlir tensor: " << i;
    params.inputs.push_back(GetHlirTensor(*tensor));
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    auto tensor = outputs[i];
    VLOG(6) << "start get output hlir tensor: " << i;
    params.outputs.push_back(GetHlirTensor(*tensor));
  }
}

void FreeDispatchParam(hlir::DispatchParam& params) {  // NOLINT
  for (auto input : params.inputs) delete input;
  for (auto output : params.outputs) delete output;
}

int64_t GetCurrentTimestap() {
  struct timeval tv;
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    VLOG(6) << "Func gettimeofday may failed, ret:" << ret;
    return 0;
  }
  int64_t totalUsec = tv.tv_usec + tv.tv_sec * 1000000;
  return totalUsec;
}

static const int64_t kMicrosToMillis = 1000;

double GetTimeCostMs(int64_t start_time, int64_t end_time) {
  return ((static_cast<double>(end_time) - static_cast<double>(start_time)) /
          static_cast<double>(kMicrosToMillis));
}

ChipType ParseChipType() {
  ChipType type = ChipType::UNKNOW;
  if (dtu::driver::DeviceManager::instance()->IsDorado()) {
    type = ChipType::DORADO;
    if (dtu::driver::DeviceManager::instance()->device_info().clusters_num ==
        2) {
      type = ChipType::DORADO_2C;
    } else {
      VLOG(1) << "[WARN] Paddle now only suport dorado_2c in dorado platform!";
    }
  } else if (dtu::driver::DeviceManager::instance()->IsScorpio()) {
    type = ChipType::SCORPIO;
  } else if (dtu::driver::DeviceManager::instance()->IsPavo()) {
    type = ChipType::PAVO;
  }
  PADDLE_ENFORCE_NE(
      type,
      ChipType::UNKNOW,
      phi::errors::Unavailable("unknown chip type is not support!"));
  return type;
}

static std::string GetChipTypeStr(ChipType type) {
  switch (type) {
    case ChipType::LEO:
      return "leo";
    case ChipType::PAVO:
      return "pavo";
    case ChipType::PAVO_1C:
      return "pavo_1c";
    case ChipType::DORADO:
      return "dorado";
    case ChipType::DORADO_2C:
      return "dorado_2c";
    case ChipType::DORADO_3PG:
      return "dorado_3pg";
    case ChipType::LIBRA:
      return "libra";
    case ChipType::SCORPIO:
      return "scorpio";
    default:
      return "unknown";
  }
}

std::string GetTargetName() { return GetChipTypeStr(ParseChipType()); }

phi::DenseTensor EmptyTensor(const phi::DeviceContext& dev_ctx,
                             const phi::DataType dtype,
                             const phi::DDim& dims,
                             const phi::DataLayout layout) {
  phi::DenseTensorMeta meta(dtype, dims, layout);
  return EmptyTensor(dev_ctx, meta);
}

phi::DenseTensor EmptyTensor(const phi::DeviceContext& dev_ctx,
                             const phi::DenseTensorMeta& meta) {
  phi::DenseTensor output_tensor;
  output_tensor.set_meta(meta);
  dev_ctx.Alloc(&output_tensor, output_tensor.dtype());
  return output_tensor;
}

}  // namespace custom_kernel
