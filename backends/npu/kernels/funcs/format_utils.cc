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
//
// Part of the following code in this file is from
//     https://gitee.com/ascend/pytorch/blob/master/torch_npu/csrc/framework/FormatHelper.cpp
//     Git commit hash: 3fc20b8b30d18e2da3f73888ae47effe5a3d1ca2 (v1.5.0)
// Retain the following license from the original files:
//
// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/format_utils.h"

#include "kernels/funcs/string_helper.h"

static std::map<phi::DataType, aclDataType>  //
    DTYPE_2_ACL_DTYPE = {
        {phi::DataType::BOOL, ACL_BOOL},
        {phi::DataType::UINT8, ACL_UINT8},
        {phi::DataType::INT8, ACL_INT8},
        {phi::DataType::INT16, ACL_INT16},
        {phi::DataType::INT32, ACL_INT32},
        {phi::DataType::INT64, ACL_INT64},
        {phi::DataType::FLOAT16, ACL_FLOAT16},
        {phi::DataType::BFLOAT16, ACL_BF16},
        {phi::DataType::FLOAT32, ACL_FLOAT},
        {phi::DataType::FLOAT64, ACL_DOUBLE},
        {phi::DataType::COMPLEX64, ACL_COMPLEX64},
        {phi::DataType::COMPLEX128, ACL_COMPLEX128},
};

static std::map<phi::DataLayout, aclFormat> DATA_LAYOUT_2_ACL_FORMAT = {
    {phi::DataLayout::NCHW, ACL_FORMAT_NCHW},
    {phi::DataLayout::NHWC, ACL_FORMAT_NHWC},
    {phi::DataLayout::kNCDHW, ACL_FORMAT_NCDHW},
    {phi::DataLayout::kNDHWC, ACL_FORMAT_NDHWC},
    {phi::DataLayout::ANY, ACL_FORMAT_ND},
};

aclDataType ConvertToNpuDtype(phi::DataType dtype) {
  auto iter = DTYPE_2_ACL_DTYPE.find(dtype);
  PADDLE_ENFORCE_NE(
      iter,
      DTYPE_2_ACL_DTYPE.end(),
      phi::errors::NotFound(
          "The data type %s can not convert to ACL data type.", dtype));
  return iter->second;
}

aclFormat ConvertToNpuFormat(phi::DataLayout layout) {
  auto iter = DATA_LAYOUT_2_ACL_FORMAT.find(layout);
  PADDLE_ENFORCE_NE(
      iter,
      DATA_LAYOUT_2_ACL_FORMAT.end(),
      phi::errors::NotFound(
          "The data type (%s) can not convert to ACL data type.", layout));
  return iter->second;
}

namespace {

constexpr int BLOCKSIZE = 16;

// base format is ND/NCHW
FormatShape InferShapeLessTo4(FormatShape dims);
FormatShape InferShape4To5(FormatShape dims);
FormatShape InferShape5To4(FormatShape dims);
FormatShape InferShapeNDToNZ(FormatShape dims);
FormatShape InferShapeNDToZ(FormatShape dims);
FormatShape InferShapeofNCHW(FormatShape dims);
FormatShape InferShapeofND(FormatShape dims);

// converter between base format
FormatShape InferShapeNCHWToND(FormatShape storage_dims, FormatShape base_dims);
FormatShape InferShapeNCDHWToND(FormatShape storage_dims,
                                FormatShape base_dims);
FormatShape InferShapeNDToNCHW(FormatShape storage_dims, FormatShape base_dims);
FormatShape InferShapeNDToNCDHW(FormatShape storage_dims,
                                FormatShape base_dims);

// base format is NCDHW
FormatShape InferShapeOfNDHWC(FormatShape dims);
FormatShape InferShapeOfNCDHW(FormatShape dims);
FormatShape InferShapeOfNDC1HWC0(FormatShape dims);
FormatShape InferShapeOfFZ3D(FormatShape dims);
}  // namespace

// clang-format off
std::unordered_map<aclFormat, FormatHelper::FormatInfo> FormatHelper::info = {
  {ACL_FORMAT_NC1HWC0,      (FormatInfo){ACL_FORMAT_NC1HWC0,    ACL_FORMAT_NCHW,    InferShape4To5,         "NC1HWC0",      true}}, // NOLINT
  {ACL_FORMAT_ND,           (FormatInfo){ACL_FORMAT_ND,         ACL_FORMAT_ND,      InferShapeofND,         "ND",           false}}, // NOLINT
  {ACL_FORMAT_NCHW,         (FormatInfo){ACL_FORMAT_NCHW,       ACL_FORMAT_NCHW,    InferShapeofNCHW,       "NCHW",         false}}, // NOLINT
  {ACL_FORMAT_FRACTAL_NZ,   (FormatInfo){ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND,      InferShapeNDToNZ,       "FRACTAL_NZ",   true}}, // NOLINT
  {ACL_FORMAT_FRACTAL_Z,    (FormatInfo){ACL_FORMAT_FRACTAL_Z,  ACL_FORMAT_NCHW,    InferShapeNDToZ,        "FRACTAL_Z",    true}}, // NOLINT
  {ACL_FORMAT_NDHWC,        (FormatInfo){ACL_FORMAT_NDHWC,      ACL_FORMAT_NCDHW,   InferShapeOfNDHWC,      "NDHWC",        false}}, // NOLINT
  {ACL_FORMAT_NCDHW,        (FormatInfo){ACL_FORMAT_NCDHW,      ACL_FORMAT_NCDHW,   InferShapeOfNCDHW,      "NCDHW",        false}}, // NOLINT
  {ACL_FORMAT_NDC1HWC0,     (FormatInfo){ACL_FORMAT_NDC1HWC0,   ACL_FORMAT_NCDHW,   InferShapeOfNDC1HWC0,   "NDC1HWC0",     true}}, // NOLINT
  {ACL_FRACTAL_Z_3D,        (FormatInfo){ACL_FRACTAL_Z_3D,      ACL_FORMAT_NCDHW,   InferShapeOfFZ3D,       "FRACTAL_Z_3D", true}}, // NOLINT
};
// clang-format on

char* FormatHelper::GetFormatName(const aclFormat& format) {
  const auto& itr = info.find(format);
  if (itr == info.end()) {
    LOG(FATAL) << "unknown format type:" << format;
    return nullptr;
  }
  return itr->second.formatName;
}

FormatShape FormatHelper::GetStorageShape(const aclFormat storage_format,
                                          const FormatShape origin_dims) {
  auto itr = info.find(storage_format);
  if (itr != info.end()) {
    if (itr->second.func) {
      return itr->second.func(origin_dims);  // change ori_size to storage_size
    }
  }
  LOG(FATAL) << "unsupport InferShape with format "
             << GetFormatName(storage_format) << "with shape"
             << GetVectorString(origin_dims);
  return {};
}

namespace {
FormatShape InferShapeLessTo4(FormatShape dims) {
  FormatShape res;
  res.resize(4);
  PADDLE_ENFORCE_LE(
      dims.size(),
      4,
      phi::errors::InvalidArgument("input dim > 4 when InferShapeLessTo4"));
  switch (dims.size()) {
    case 0:
      res[0] = 1;
      res[1] = 1;
      res[2] = 1;
      res[3] = 1;
      break;
    case 1:
      // RESHAPE_TYPE_C;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = 1;
      res[3] = 1;
      break;
    case 2:
      // RESHAPE_TYPE_CH;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = dims[1];
      res[3] = 1;
      break;
    case 3:
      // RESHAPE_TYPE_CHW;
      res[0] = 1;
      res[1] = dims[0];
      res[2] = dims[1];
      res[3] = dims[2];
      break;
    case 4:
      res[0] = dims[0];
      res[1] = dims[1];
      res[2] = dims[2];
      res[3] = dims[3];
      break;
    default:
      LOG(FATAL) << "dims of NCHW shape should not be greater than 4, which is "
                 << dims.size();
  }
  return res;
}

FormatShape InferShape4To5(FormatShape dims) {
  FormatShape res;
  res.resize(5);
  if (dims.size() < 4) {
    VLOG(4) << "infershape4to5 but input dim < 4";
    return InferShape4To5(InferShapeLessTo4(dims));
  } else if (dims.size() > 4) {
    VLOG(4) << "infershape4to5 but input dim > 4";
  }
  res[0] = dims[0];
  res[1] = (dims[1] + 15) / 16;
  res[2] = dims[2];
  res[3] = dims[3];
  res[4] = BLOCKSIZE;
  return res;
}

FormatShape InferShape5To4(FormatShape dims) {
  FormatShape res;
  res.emplace_back(dims[0]);
  res.emplace_back(((dims[1] + 15) / 16) * 16);
  res.emplace_back(dims[2]);
  res.emplace_back(dims[3]);
  return res;
}

FormatShape InferShapeNDToNZ(FormatShape dims) {
  FormatShape res;
  // sum(keepdim = false) may make tensor dim = 0
  FormatShape dim;
  for (int i = 0; i < dims.size(); i++) {
    dim.emplace_back(dims[i]);
  }

  // TODO(ascend): this expand code can be remove now
  // this action will move to GuessStorageSizeWhenConvertFormat
  if (dim.size() == 0) {
    dim.emplace_back(1);
  }
  if (dim.size() == 1) {
    dim.emplace_back(1);
  }

  int i = 0;
  for (; i < dim.size() - 2; i++) {
    res.emplace_back(dim[i]);
  }

  res.emplace_back((dim[i + 1] + 15) / BLOCKSIZE);
  res.emplace_back((dim[i] + 15) / BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);

  return res;
}

FormatShape InferShapeNDToZ(FormatShape dims) {
  FormatShape res;
  if (dims.size() < 4) {
    return InferShapeNDToZ(InferShapeLessTo4(dims));
  }

  res.emplace_back((dims[1] + 15) / BLOCKSIZE * dims[2] * dims[3]);
  res.emplace_back((dims[0] + 15) / BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);

  return res;
}

FormatShape InferShapeNCHWToND(FormatShape storage_dims,
                               FormatShape base_dims) {
  FormatShape res;
  res.resize(4);
  auto cur_storage_dims = storage_dims;
  if (storage_dims.size() != 4) {
    cur_storage_dims = InferShapeLessTo4(storage_dims);
  }
  PADDLE_ENFORCE_EQ(cur_storage_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "input dim num not equal 4 when InferShapeNCHWToND"));

  if (base_dims.size() == 0) {
    FormatShape temp_dims;
    temp_dims.emplace_back(1);
    return InferShapeLessTo4(temp_dims);
  }
  switch (base_dims.size()) {
    case 1:
      // reshape_type = RESHAPE_TYPE_C;
      res.resize(1);
      res[0] = cur_storage_dims[1];
      PADDLE_ENFORCE_EQ(
          cur_storage_dims[0],
          1,
          phi::errors::InvalidArgument(
              "reshape type RESHAPE_TYPE_C erase dim N must be 1"));
      PADDLE_ENFORCE_EQ(
          cur_storage_dims[2],
          1,
          phi::errors::InvalidArgument(
              "reshape type RESHAPE_TYPE_C erase dim H must be 1"));
      PADDLE_ENFORCE_EQ(
          cur_storage_dims[3],
          1,
          phi::errors::InvalidArgument(
              "reshape type RESHAPE_TYPE_C erase dim W must be 1"));
      break;
    case 2:
      // reshape_type = RESHAPE_TYPE_CH;
      res.resize(2);
      res[0] = cur_storage_dims[1];
      res[1] = cur_storage_dims[2];
      PADDLE_ENFORCE_EQ(
          cur_storage_dims[0],
          1,
          phi::errors::InvalidArgument(
              "reshape type RESHAPE_TYPE_CH erase dim N must be 1"));
      PADDLE_ENFORCE_EQ(
          cur_storage_dims[3],
          1,
          phi::errors::InvalidArgument(
              "reshape type RESHAPE_TYPE_CH erase dim W must be 1"));
      break;
    case 3:
      // reshape_type = RESHAPE_TYPE_CHW;
      res.resize(3);
      res[0] = cur_storage_dims[1];
      res[1] = cur_storage_dims[2];
      res[2] = cur_storage_dims[3];
      PADDLE_ENFORCE_EQ(
          cur_storage_dims[0],
          1,
          phi::errors::InvalidArgument(
              "reshape type RESHAPE_TYPE_CHW erase dim N must be 1"));
      break;
    case 4:
      res = cur_storage_dims;
      return res;
    default:
      LOG(FATAL) << "unknown reshape type:";
  }
  return res;
}

FormatShape InferShapeNDToNCHW(FormatShape storage_dims,
                               FormatShape base_dims) {
  PADDLE_ENFORCE_LE(
      storage_dims.size(),
      4,
      phi::errors::InvalidArgument("input storage dim not less than 4"));
  PADDLE_ENFORCE_LE(
      base_dims.size(),
      4,
      phi::errors::InvalidArgument("input storage dim not less than 4"));
  return InferShapeLessTo4(base_dims);
}

FormatShape InferShapeNDToNCDHW(FormatShape storage_dims,
                                FormatShape base_dims) {
  PADDLE_ENFORCE_EQ(
      storage_dims.size(),
      5,
      phi::errors::InvalidArgument("ND failed to convert to NCDHW"));
  FormatShape res;
  res.resize(5);
  res = storage_dims;
  return res;
}

FormatShape InferShapeNCDHWToND(FormatShape storage_dims,
                                FormatShape base_dims) {
  FormatShape res;
  res.resize(5);
  res = storage_dims;
  PADDLE_ENFORCE_EQ(res.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "input dim num not equal 5 when InferShapeNCDHWToND"));
  return res;
}

// NCDHW -> NDHWC
FormatShape InferShapeOfNDHWC(FormatShape dims) {
  if (dims.size() < 5) {
    LOG(FATAL) << "dim (", dims, ") cannot convert to NDHWC";
  }
  FormatShape res;
  res.resize(5);
  res[0] = dims[0];
  res[1] = dims[2];
  res[2] = dims[3];
  res[3] = dims[4];
  res[4] = dims[1];
  return res;
}

// NCDHW to NCDHW
FormatShape InferShapeOfNCDHW(FormatShape dims) {
  if (dims.size() < 5) {
    LOG(FATAL) << "dim (", dims, ") cannot convert to NCDHW";
  }
  FormatShape res;
  res.resize(5);
  res[0] = dims[0];
  res[1] = dims[1];
  res[2] = dims[2];
  res[3] = dims[3];
  res[4] = dims[4];
  return res;
}

// NCDHW to NDC1HWC0
FormatShape InferShapeOfNDC1HWC0(FormatShape dims) {
  if (dims.size() < 5) {
    LOG(FATAL) << "dim (", dims, ") cannot convert to NDC1HWC0";
  }
  FormatShape res;
  res.resize(6);
  res[0] = dims[0];
  res[1] = dims[2];
  res[2] = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
  res[3] = dims[3];
  res[4] = dims[4];
  res[5] = BLOCKSIZE;
  return res;
}

// NCDHW to FZ_3D
FormatShape InferShapeOfFZ3D(FormatShape dims) {
  if (dims.size() < 5) {
    LOG(FATAL) << "dim (", dims, ") cannot convert to FZ_3D";
  }

  int64_t d1 = dims[2];
  int64_t d2 = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
  int64_t d3 = dims[3];
  int64_t d4 = dims[4];
  int64_t d5 = (dims[0] + BLOCKSIZE - 1) / BLOCKSIZE;
  int64_t d6 = BLOCKSIZE;
  int64_t d7 = BLOCKSIZE;

  // The shape of FZ3D is 7D, but the CANN only accept 4D
  // so we should merge 1st, 2nd, 3rd, 4th dimension.
  FormatShape res;
  res.resize(4);
  res[0] = d1 * d2 * d3 * d4;
  res[1] = d5;
  res[2] = d6;
  res[3] = d7;
  return res;
}

FormatShape InferShapeofNCHW(FormatShape dims) {
  return InferShapeLessTo4(dims);
}

FormatShape InferShapeofND(FormatShape dims) {
  FormatShape res;
  res.resize(dims.size());
  for (int j = 0; j < dims.size(); j++) {
    res[j] = dims[j];
  }
  return res;
}
}  // namespace