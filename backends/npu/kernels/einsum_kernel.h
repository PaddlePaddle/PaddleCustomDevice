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
#pragma once

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

enum LabelType {
  ALL_TYPE = 0,
  Batch = 1,    // ABO
  AO,           // AO --  free label
  BO,           // BO --  free label
  Contraction,  // AB
  Reduction,    // A, B
};

// map a label('a' - 'z') -> int, O(1) speed.
class LabelMap {
  constexpr static int N =
      26 + 1;  // 'a' - 'z' + '.', '.' is for broadcast dims
  int default_value;
  int map[N];

 public:
  explicit LabelMap(int default_value = 0) {
    this->default_value = default_value;
    for (int i = 0; i < N; ++i) map[i] = default_value;
  }
  int& operator[](int label) {
    int i = label - 'a';
    if (label == '.') i = N - 1;
    return map[i];
  }
  int operator[](int label) const {
    int i = label - 'a';
    if (label == '.') i = N - 1;
    return map[i];
  }
  bool exist(char label) { return !is_default(label); }

 private:
  // non-exist is present by is_default
  bool is_default(char label) {
    return (*this)[static_cast<int>(label)] == default_value;
  }
};

// Transpose Kernel
template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out);

template <typename T, typename Context>
phi::DenseTensor Transpose(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const std::vector<int>& axis) {
  phi::DenseTensor out;
  // infer out shape
  {
    auto x_dims = x.dims();
    int x_rank = x_dims.size();
    int axis_size = axis.size();

    // Note: x_rank > axis_size when fuse squeeze2 + transpose2, else x_rank ==
    // axis_size
    PADDLE_ENFORCE_GE(x_rank,
                      axis_size,
                      phi::errors::InvalidArgument(
                          "The input tensor's dimension "
                          "should be equal to or greater than the axis's size. "
                          "But received input tensor's dimension is %d, "
                          "axis's size is %d",
                          x_rank,
                          axis_size));

    std::vector<int> formated_axis = axis;
    std::vector<int> count(axis_size, 0);
    for (int i = 0; i < axis_size; i++) {
      PADDLE_ENFORCE_LT(axis[i],
                        x_rank,
                        phi::errors::InvalidArgument(
                            "The reduce dim index %d should be in the "
                            "range [ -dimension(X), dimension(X) ) "
                            "which dimesion = %d. But received dim index = %d.",
                            i,
                            x_rank,
                            axis[i]));
      PADDLE_ENFORCE_GE(axis[i],
                        -x_rank,
                        phi::errors::InvalidArgument(
                            "The reduce dim index %d should be in the "
                            "range [ -dimension(X), dimension(X) )  "
                            "which dimesion = %d. But received dim index = %d.",
                            i,
                            x_rank,
                            axis[i]));

      if (axis[i] < 0) {
        formated_axis[i] = axis[i] + x_rank;
      }
      PADDLE_ENFORCE_EQ(++count[formated_axis[i]],
                        1,
                        phi::errors::InvalidArgument(
                            "Each element of axis should be unique. but "
                            "axis[%d] is %d appear not only once",
                            i,
                            axis[i]));
    }

    phi::DDim out_dims(x_dims);
    for (int i = 0; i < axis_size; ++i) {
      out_dims[i] = x_dims[formated_axis[i]];
    }
    phi::DenseTensorMeta out_meta = {x.dtype(), out_dims};
    out.set_meta(out_meta);
  }
  custom_kernel::TransposeKernel<T, Context>(dev_ctx, x, axis, &out);
  return out;
}

// DiagonalKernel
template <typename T, typename Context>
void DiagonalKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    phi::DenseTensor* out);

template <typename T, typename Context>
phi::DenseTensor Diagonal(const Context& dev_ctx,
                          const phi::DenseTensor& input,
                          int offset,
                          int axis1,
                          int axis2) {
  phi::DenseTensor out;
  // infer out shape
  {
    auto x_dims = input.dims();
    int offset_ = offset;
    int axis1_ = axis1 < 0 ? x_dims.size() + axis1 : axis1;
    int axis2_ = axis2 < 0 ? x_dims.size() + axis2 : axis2;
    PADDLE_ENFORCE_GE(
        x_dims.size(),
        2,
        phi::errors::OutOfRange("Input's dim is out of range (expected at "
                                "least 2 dimensions, but got %ld).",
                                x_dims.size()));
    PADDLE_ENFORCE_LT(
        axis1_,
        x_dims.size(),
        phi::errors::OutOfRange(
            "Attr(axis1) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()),
            (x_dims.size() - 1),
            axis1));
    PADDLE_ENFORCE_GE(
        axis1_,
        0,
        phi::errors::OutOfRange(
            "Attr(axis1) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()),
            (x_dims.size() - 1),
            axis1));
    PADDLE_ENFORCE_LT(
        axis2_,
        x_dims.size(),
        phi::errors::OutOfRange(
            "Attr(axis2) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()),
            (x_dims.size() - 1),
            axis2));
    PADDLE_ENFORCE_GE(
        axis2_,
        0,
        phi::errors::OutOfRange(
            "Attr(axis2) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()),
            (x_dims.size() - 1),
            axis2));
    PADDLE_ENFORCE_NE(
        axis1_,
        axis2_,
        phi::errors::InvalidArgument("The dimensions should not be identical "
                                     "%d vs %d.",
                                     axis1,
                                     axis2));

    auto out_dims = phi::vectorize(x_dims);
    // from out_dims get the dim size of axis1_.
    auto axis1_size = out_dims[axis1_];
    auto axis2_size = out_dims[axis2_];
    // delete two dims by attr axis1 and axis2 from out_dims.
    /* example:
        out_dim = [2, 3, 4];
        axis1 = 0;
        axis2 = 1;
        according to the attr of axis1 and axis2, we get:
        out_dim = [4].
    */
    out_dims.erase(out_dims.begin() + std::max(axis1_, axis2_));
    out_dims.erase(out_dims.begin() + std::min(axis1_, axis2_));

    if (offset_ == 0) {
      out_dims.push_back(std::min(axis1_size, axis2_size));
    } else if (offset_ > 0) {
      if ((axis2_size - offset_) > 0) {
        out_dims.push_back(std::min(axis1_size, axis2_size - offset_));
      } else {
        out_dims.push_back(0);
      }
    } else {
      if ((axis1_size + offset_) > 0) {
        out_dims.push_back(std::min(axis1_size + offset_, axis2_size));
      } else {
        out_dims.push_back(0);
      }
    }
    auto out_dtype = input.dtype();
    phi::DenseTensorMeta out_meta = {out_dtype, phi::make_ddim(out_dims)};
    out.set_meta(out_meta);
  }
  custom_kernel::DiagonalKernel<T, Context>(
      dev_ctx, input, offset, axis1, axis2, &out);
  return out;
}

// FillDiagonalKernel
template <typename T, typename Context>
void FillDiagonalTensorKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& y,
                              int64_t offset,
                              int dim1,
                              int dim2,
                              phi::DenseTensor* out);

template <typename T, typename Context>
phi::DenseTensor FillDiagonalTensor(const Context& dev_ctx,
                                    const phi::DenseTensor& x,
                                    const phi::DenseTensor& y,
                                    int64_t offset,
                                    int dim1,
                                    int dim2) {
  phi::DenseTensor out;
  phi::DenseTensorMeta out_meta = {x.dtype(), x.dims()};
  out.set_meta(out_meta);
  custom_kernel::FillDiagonalTensorKernel<T, Context>(
      dev_ctx, x, y, offset, dim1, dim2, &out);
  return out;
}

// Sum kernel
template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
phi::DenseTensor Sum(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::IntArray& axis,
                     phi::DataType dtype,
                     bool keep_dim) {
  phi::DenseTensor dense_out;
  // infer shape
  {
    bool reduce_all = false;
    if (axis.size() == 0) {
      reduce_all = true;
    }
    phi::DDim out_dim;
    if (!axis.FromTensor()) {
      auto x_rank = x.dims().size();
      std::vector<int64_t> formated_axis = axis.GetData();
      for (size_t i = 0; i < axis.size(); ++i) {
        if (x_rank == 0) {
          PADDLE_ENFORCE_EQ(
              axis[i] == 0 || axis[i] == -1,
              true,
              phi::errors::InvalidArgument("When input 0D Tensor, the axis can "
                                           "only be -1, 0, None or []"));
        } else {
          PADDLE_ENFORCE_LT(
              axis[i],
              x_rank,
              phi::errors::InvalidArgument(
                  "The reduce dim index %d should be in the "
                  "range [ -dimension(X), dimension(X) ) "
                  "which dimesion = %d. But received dim index = %d.",
                  i,
                  x_rank,
                  axis[i]));
          PADDLE_ENFORCE_GE(
              axis[i],
              -x_rank,
              phi::errors::InvalidArgument(
                  "The reduce dim index %d should be in the "
                  "range [ -dimension(X), dimension(X) )  "
                  "which dimesion = %d. But received dim index = %d.",
                  i,
                  x_rank,
                  axis[i]));
        }

        if (axis[i] < 0) {
          formated_axis[i] = axis[i] + x_rank;
        }
      }

      bool full_dim = true;
      std::set<int64_t> dims_set(formated_axis.begin(), formated_axis.end());
      for (int64_t i = 0; i < x_rank; ++i) {
        if (dims_set.find(i) == dims_set.end()) {
          full_dim = false;
          break;
        }
      }
      reduce_all = reduce_all || full_dim;

      std::vector<int64_t> out_dim_vector;
      for (int64_t i = 0; i < x_rank; ++i) {
        if (reduce_all || dims_set.find(i) != dims_set.end()) {
          if (keep_dim) {
            out_dim_vector.push_back(1);
          } else {
            continue;
          }
        } else {
          out_dim_vector.push_back(x.dims().at(i));
        }
      }

      out_dim = phi::make_ddim(out_dim_vector);
    } else {
      std::vector<int64_t> vec_axis = axis.GetData();
      std::vector<int64_t> vec_dim;
      if (reduce_all) {
        if (keep_dim) {
          vec_dim = std::vector<int64_t>(x.dims().size(), 1);
        } else {
          vec_dim = {};
        }
      } else {
        if (keep_dim) {
          vec_dim = std::vector<int64_t>(x.dims().size(), -1);
        } else {
          auto x_rank = static_cast<size_t>(x.dims().size());
          if (vec_axis.size() > x_rank) {
            vec_dim = {-1};
          } else {
            vec_dim =
                std::vector<int64_t>(x.dims().size() - vec_axis.size(), -1);
          }
        }
      }
      out_dim = phi::make_ddim(vec_dim);
    }

    phi::DataType out_dtype;
    if (dtype != phi::DataType::UNDEFINED) {
      out_dtype = dtype;
    } else {
      if (x.dtype() == phi::DataType::BOOL ||
          x.dtype() == phi::DataType::INT32) {
        out_dtype = phi::DataType::INT64;
      } else {
        out_dtype = x.dtype();
      }
    }

    phi::DenseTensorMeta out_meta = {out_dtype, out_dim, x.layout()};
    dense_out.set_meta(out_meta);
  }
  custom_kernel::SumKernel<T, Context>(
      dev_ctx, x, axis, dtype, keep_dim, &dense_out);
  return dense_out;
}

// Matmul kernel

template <typename T, typename Context>
void MatmulKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out);

template <typename T, typename Context>
phi::DenseTensor Matmul(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        bool transpose_x = false,
                        bool transpose_y = false) {
  phi::DenseTensor dense_out;
  // infer shape
  {
    std::vector<int64_t> dims_x = phi::vectorize(x.dims());
    std::vector<int64_t> dims_y = phi::vectorize(y.dims());
    auto ndims_x = dims_x.size();
    auto ndims_y = dims_y.size();
    PADDLE_ENFORCE_GT(ndims_x,
                      0UL,
                      phi::errors::InvalidArgument(
                          "The Input(x) dims size must be greater than 0,"
                          " but reviced dims size is 0. "));
    PADDLE_ENFORCE_GT(ndims_y,
                      0UL,
                      phi::errors::InvalidArgument(
                          "The Input(y) dims size must be greater than 0,"
                          " but reviced dims size is 0. "));

    bool x_broadcasted = false, y_broadcasted = false;
    if (ndims_x == 1) {
      dims_x.insert(dims_x.begin(), 1);
      ndims_x = 2;
      x_broadcasted = true;
    }

    if (ndims_y == 1) {
      dims_y.push_back(1);
      ndims_y = 2;
      y_broadcasted = true;
    }

    size_t M, N;
    if (transpose_x) {
      M = dims_x[ndims_x - 1];
    } else {
      M = dims_x[ndims_x - 2];
    }
    if (transpose_y) {
      N = dims_y[ndims_y - 2];
    } else {
      N = dims_y[ndims_y - 1];
    }

    std::vector<int64_t> new_dims;
    if (ndims_x > ndims_y) {
      new_dims.assign(dims_x.begin(), dims_x.end() - 2);
    } else if (ndims_x < ndims_y) {
      new_dims.assign(dims_y.begin(), dims_y.end() - 2);
    } else {
      new_dims.reserve(ndims_x);
      for (size_t i = 0; i < ndims_x - 2; ++i) {
        new_dims.push_back(std::max(dims_x[i], dims_y[i]));
      }
    }
    if (!x_broadcasted) {
      new_dims.push_back(M);
    }
    if (!y_broadcasted) {
      new_dims.push_back(N);
    }

    auto ddim_out = phi::make_ddim(new_dims);

    phi::DenseTensorMeta out_meta = {x.dtype(), ddim_out, x.layout()};
    dense_out.set_meta(out_meta);
  }
  custom_kernel::MatmulKernel<T, Context>(
      dev_ctx, x, y, transpose_x, transpose_y, &dense_out);
  return dense_out;
}

// Full kernel
template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
phi::DenseTensor Full(const Context& dev_ctx,
                      const phi::IntArray& shape,
                      const phi::Scalar& val) {
  phi::DenseTensor dense_out;
  auto dtype = phi::CppTypeToDataType<T>::Type();
  phi::DenseTensorMeta out_meta = {phi::CppTypeToDataType<T>::Type(),
                                   phi::make_ddim(shape.GetData()),
                                   phi::DataLayout::NCHW};
  dense_out.set_meta(out_meta);
  // infer shape
  custom_kernel::FullKernel<T, Context>(dev_ctx, shape, val, dtype, &dense_out);
  return dense_out;
}

// Tile kernel
template <typename T, typename Context>
void TileKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& repeat_times,
                phi::DenseTensor* out);

// If T is not complex
template <
    typename T,
    typename Context,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
phi::DenseTensor Conj(const Context& dev_ctx UNUSED,
                      const phi::DenseTensor& x) {
  return x;
}

}  // namespace custom_kernel
