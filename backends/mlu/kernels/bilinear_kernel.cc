// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/elementwise_utils.h"
#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "kernels/funcs/reduce_op.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace custom_kernel {
template <typename T, typename Context>
void SetTensorValueKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& value,
                          const phi::IntArray& starts,
                          const phi::IntArray& ends,
                          const phi::IntArray& steps,
                          const std::vector<int64_t>& axes,
                          const std::vector<int64_t>& decrease_axes,
                          const std::vector<int64_t>& none_axes,
                          phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();

  auto in_dims = x.dims();
  phi::funcs::CheckAndUpdateSliceAttrs(
      in_dims, axes, &starts_local, &ends_local, &steps_local);
  auto slice_dims = phi::funcs::GetSliceDims(
      in_dims, axes, starts_local, ends_local, &steps_local);
  auto decrease_slice_dims =
      phi::funcs::GetDecreasedDims(slice_dims, decrease_axes);

  auto slice_dims_for_assign = decrease_slice_dims;
  if (!none_axes.empty()) {
    std::vector<int64_t> slice_dims_with_none;
    size_t none_axes_cur = 0, decrease_axes_cur = 0;
    for (int i = 0; i < slice_dims.size(); ++i) {
      while (none_axes_cur < none_axes.size() &&
             none_axes[none_axes_cur] <= i) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }
      if (decrease_axes_cur < decrease_axes.size() &&
          decrease_axes[decrease_axes_cur] == i) {
        decrease_axes_cur++;
      } else {
        slice_dims_with_none.push_back(slice_dims[i]);
      }
    }
    while (none_axes_cur < none_axes.size()) {
      slice_dims_with_none.push_back(1);
      none_axes_cur++;
    }

    slice_dims_for_assign = phi::make_ddim(slice_dims_with_none);
  }
  int in_size = in_dims.size();
  int64_t starts_indices[in_size] = {0};
  int64_t ends_indices[in_size] = {0};
  int64_t strides_indices[in_size] = {0};

  for (int64_t i = 0; i < in_dims.size(); ++i) {
    starts_indices[i] = 0;
    ends_indices[i] = static_cast<int64_t>(slice_dims[i]);
    strides_indices[i] = 1;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int axis_index = axes[i];
    starts_indices[axis_index] = static_cast<int64_t>(starts_local[i]);
    ends_indices[axis_index] = static_cast<int64_t>(ends_local[i]);
    strides_indices[axis_index] = static_cast<int64_t>(steps_local[i]);
  }

  phi::DenseTensor value_temp;
  if (slice_dims_for_assign == value.dims()) {
    value_temp = value;
  } else {
    value_temp.Resize(slice_dims_for_assign);
    dev_ctx.template Alloc<T>(&value_temp);
    MLUCnnlTensorDesc value_t_desc(value);
    MLUCnnlTensorDesc value_temp_desc(value_temp);
    MLUCnnl::BroadcastTo(dev_ctx,
                         value_t_desc.get(),
                         GetBasePtr(&value),
                         value_temp_desc.get(),
                         GetBasePtr(&value_temp));
  }

  int64_t input_numel = phi::product(in_dims);
  int64_t value_numel = phi::product(value_temp.dims());
  phi::DenseTensor in_temp, out_temp, val_temp, index_out;
  int64_t stride_step = phi::product(in_dims);
  std::vector<int64_t> index_indices(stride_step);
  std::iota(index_indices.begin(), index_indices.end(), 0);
  phi::DenseTensor index_temp;
  in_temp = x;
  val_temp = value_temp;
  custom_kernel::TensorFromVector(dev_ctx, index_indices, dev_ctx, &index_temp);
  dev_ctx.Wait();
  index_temp.Resize(in_dims);
  auto index_dims = in_dims;
  for (int64_t i = 0; i < in_dims.size(); ++i) {
    if (starts_indices[i] < 0 || ends_indices[i] < 0) {
      starts_indices[i] -= in_dims[i];
      ends_indices[i] -= in_dims[i];
    }
    if (strides_indices[i] > 0)
      index_dims[i] =
          static_cast<int64_t>((ends_indices[i] - starts_indices[i] - 1) /
                               strides_indices[i]) +
          1;
    else
      index_dims[i] =
          static_cast<int64_t>((ends_indices[i] - starts_indices[i] + 1) /
                               strides_indices[i]) +
          1;
  }
  auto new_in_dims = phi::make_ddim({input_numel});
  auto new_val_dims = phi::make_ddim({value_numel});
  in_temp.Resize(new_in_dims);
  val_temp.Resize(new_val_dims);
  index_out.Resize(index_dims);
  dev_ctx.template Alloc<int64_t>(&index_out);
  cnnlScatterRefMode_t mode = CNNL_SCATTERREF_UPDATE;
  MLUCnnlTensorDesc x_desc(in_temp);
  MLUCnnlTensorDesc indices_desc(index_temp);
  MLUCnnlTensorDesc indices_out_desc(index_out);
  MLUCnnlTensorDesc updates_desc(val_temp);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::StridedSlice(dev_ctx,
                        starts_indices,
                        ends_indices,
                        strides_indices,
                        indices_desc.get(),
                        GetBasePtr(&index_temp),
                        indices_out_desc.get(),
                        GetBasePtr(&index_out));
  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(phi::product(index_out.dims())),
      phi::product(slice_dims_for_assign),
      phi::errors::InvalidArgument(
          "OP(set_value) error index indices and value update not match "));
  // When input x and input value's dtype is int64, cast datatype to
  // int32 for cnnl ScatterRef usage
  Tensor Ref, Update;
  Ref.Resize(in_temp.dims());
  Update.Resize(val_temp.dims());

  if (in_temp.dtype() != DataType::INT64 &&
      val_temp.dtype() != DataType::INT64 &&
      in_temp.dtype() != DataType::BOOL && val_temp.dtype() != DataType::BOOL) {
    Ref = in_temp;
    Update = val_temp;
  } else {
    dev_ctx.template Alloc<int32_t>(&Ref);
    dev_ctx.template Alloc<int32_t>(&Update);
    MLUCnnlTensorDesc in_temp_desc(in_temp);
    MLUCnnlTensorDesc Ref_desc(Ref);
    MLUCnnlTensorDesc val_temp_desc(val_temp);
    MLUCnnlTensorDesc Update_desc(Update);
    cnnlCastDataType_t cast_type = GetCastDataType(x.dtype(), DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  in_temp_desc.get(),
                  GetBasePtr(&in_temp),
                  Ref_desc.get(),
                  GetBasePtr(&Ref));
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  val_temp_desc.get(),
                  GetBasePtr(&val_temp),
                  Update_desc.get(),
                  GetBasePtr(&Update));
  }
  MLUCnnlTensorDesc Ref_desc(Ref);
  MLUCnnlTensorDesc Update_desc(Update);
  Tensor index_final;
  index_final = index_out;
  int64_t indices_numel = phi::product(index_dims);
  auto new_index_dims = phi::make_ddim({indices_numel});
  index_final.Resize(new_index_dims);
  MLUCnnlTensorDesc indices_final_desc(index_final);
  MLUCnnl::ScatterRefFunctor(dev_ctx,
                             Ref_desc.get(),
                             GetBasePtr(&Ref),
                             Update_desc.get(),
                             GetBasePtr(&Update),
                             indices_final_desc.get(),
                             GetBasePtr(&index_final),
                             mode);
  Ref.Resize(in_dims);

  // When input x and input value's dtype is int64,
  // cast ScatterRef output datatype from int32 to int64
  Tensor in_temp_out;
  in_temp_out.Resize(Ref.dims());
  if (x.dtype() != DataType::INT64 && x.dtype() != DataType::BOOL) {
    in_temp_out = Ref;
  } else {
    dev_ctx.template Alloc<T>(&in_temp_out);
    MLUCnnlTensorDesc in_temp_desc(Ref);
    MLUCnnlTensorDesc in_temp_out_desc(in_temp_out);
    cnnlCastDataType_t cast_type = GetCastDataType(Ref.dtype(), x.dtype());
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  in_temp_desc.get(),
                  GetBasePtr(&Ref),
                  in_temp_out_desc.get(),
                  GetBasePtr(&in_temp_out));
  }
  TensorCopy(dev_ctx, in_temp_out, false, out);

  if (GetBasePtr(&x) != GetBasePtr(out)) {
    // a workaround method to avoid output incorrection since the op creates a
    // tensor while not using it in static graph.
    auto x_rm_const = const_cast<phi::DenseTensor&>(x);
    TensorCopy(dev_ctx, *out, false, &x_rm_const);
  }
}

static void StridedSliceOutDims(const std::vector<int64_t>& starts,
                                const std::vector<int64_t>& ends,
                                const std::vector<int64_t>& strides,
                                const std::vector<int>& axes,
                                const std::vector<int>& infer_flags,
                                const phi::DDim in_dims,
                                const std::vector<int>& decrease_axis,
                                int64_t* out_dims_vector,
                                const size_t size,
                                bool infer_shape) {
  for (int i = 0; i < in_dims.size(); i++) {
    out_dims_vector[i] = in_dims[i];
  }
  int64_t stride_index, start_index, end_index;
  for (size_t i = 0; i < size; i++) {
    int axes_index = axes[i];
    start_index = starts[i];
    end_index = ends[i];
    stride_index = strides[i];
    bool decrease_axis_affect = false;
    if (start_index == -1 && end_index == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    if (decrease_axis_affect) {
      out_dims_vector[axes_index] = 1;
      continue;
    }
    if (infer_shape && infer_flags[i] == -1) {
      out_dims_vector[axes_index] = -1;
      continue;
    }

    PADDLE_ENFORCE_NE(stride_index,
                      0,
                      phi::errors::InvalidArgument(
                          "stride index in StridedSlice operator is 0."));
    int64_t axis_size = in_dims[axes_index];

    if (axis_size < 0) {
      continue;
    }

    if (start_index < 0) {
      start_index = start_index + axis_size;
      start_index = std::max<int64_t>(start_index, 0);
    }
    if (end_index < 0) {
      if (!(end_index == -1 && stride_index < 0)) {  // skip None stop condition
        end_index = end_index + axis_size;
        if (end_index < 0) {
          end_index = 0;
        }
      }
    }

    if (stride_index < 0) {
      start_index = start_index + 1;
      end_index = end_index + 1;
    }

    bool neg_dim_condition = ((stride_index < 0 && (start_index < end_index)) ||
                              (stride_index > 0 && (start_index > end_index)));
    PADDLE_ENFORCE_EQ(neg_dim_condition,
                      false,
                      phi::errors::InvalidArgument(
                          "The start index and end index are invalid for their "
                          "corresponding stride."));

    int64_t left =
        std::max(static_cast<int64_t>(0), std::min(start_index, end_index));
    int64_t right = std::min(axis_size, std::max(start_index, end_index));
    int64_t step = std::abs(stride_index);

    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
}

static void StridedSliceFunctor(int64_t* starts,
                                int64_t* ends,
                                int64_t* strides,
                                const int* axes,
                                int* reverse_axis,
                                const phi::DDim dims,
                                const std::vector<int>& infer_flags,
                                const std::vector<int>& decrease_axis,
                                const size_t size) {
  for (size_t axis = 0; axis < size; axis++) {
    int64_t axis_size = dims[axes[axis]];
    int axis_index = axis;
    if (axis_size < 0) {
      starts[axis_index] = 0;
      ends[axis_index] = 1;
      strides[axis_index] = 1;
    }
    bool decrease_axis_affect = false;
    if (starts[axis_index] == -1 && ends[axis_index] == 0 &&
        infer_flags[axis_index] == -1) {
      auto ret = std::find(
          decrease_axis.begin(), decrease_axis.end(), axes[axis_index]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    // stride must not be zero
    if (starts[axis_index] < 0) {
      starts[axis_index] = starts[axis_index] + axis_size;
      starts[axis_index] = std::max<int64_t>(starts[axis_index], 0);
    }
    if (ends[axis_index] < 0) {
      if (!(ends[axis_index] == -1 &&
            strides[axis_index] < 0)) {  // skip None stop condition
        ends[axis_index] = ends[axis_index] + axis_size;
        if (ends[axis_index] < 0) {
          ends[axis_index] = 0;
        }
      }
    }
    if (decrease_axis_affect) {
      if (strides[axis_index] < 0) {
        ends[axis_index] = starts[axis_index] - 1;
      } else {
        ends[axis_index] = starts[axis_index] + 1;
      }
    }

    if (strides[axis_index] < 0) {
      reverse_axis[axis_index] = 1;
      strides[axis_index] = -strides[axis_index];
      if (starts[axis_index] > ends[axis_index]) {
        // swap the reverse
        auto end_dim = axis_size - 1 < starts[axis_index] ? axis_size - 1
                                                          : starts[axis_index];
        auto offset = (end_dim - ends[axis_index]) % strides[axis_index];
        offset = offset == 0 ? strides[axis_index] : offset;

        starts[axis_index] = starts[axis_index] + offset;
        ends[axis_index] = ends[axis_index] + offset;
      }
      std::swap(starts[axis_index], ends[axis_index]);
    } else {
      reverse_axis[axis_index] = 0;
      strides[axis_index] = strides[axis_index];
    }
  }
}

template <typename T, typename Context, size_t D>
void StridedSliceCompute(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const std::vector<int>& axes,
                         const phi::IntArray& starts_array,
                         const phi::IntArray& ends_array,
                         const phi::IntArray& strides_array,
                         const std::vector<int>& infer_flags,
                         const std::vector<int>& decrease_axis,
                         phi::DenseTensor* out) {
  auto in_dims = x.dims();
  // list<int>
  auto starts = starts_array.GetData();
  auto ends = ends_array.GetData();
  auto strides = strides_array.GetData();

  // out dims calculation
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  custom_kernel::StridedSliceOutDims(starts,
                                     ends,
                                     strides,
                                     axes,
                                     infer_flags,
                                     in_dims,
                                     decrease_axis,
                                     out_dims_vector.data(),
                                     axes.size(),
                                     false);
  phi::DDim out_dims(phi::make_ddim(out_dims_vector));

  // check whether need to reverse (false: stride > 0; true: stride < 0)
  std::vector<int> reverse_vector(starts.size(), 0);
  custom_kernel::StridedSliceFunctor(starts.data(),
                                     ends.data(),
                                     strides.data(),
                                     axes.data(),
                                     reverse_vector.data(),
                                     in_dims,
                                     infer_flags,
                                     decrease_axis,
                                     starts.size());

  // construct the starts_indices, ends_indices and strides_indices tensor for
  // calling StridedSlice op
  std::vector<int64_t> starts_indices_vector(D, 0);
  std::vector<int64_t> ends_indices_vector(out_dims_vector.begin(),
                                           out_dims_vector.end());
  std::vector<int64_t> strides_indices_vector(D, 1);

  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices_vector[axis_index] = starts[axis];
    ends_indices_vector[axis_index] = ends[axis];
    strides_indices_vector[axis_index] = strides[axis];
  }

  auto out_dims_origin = out_dims;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          out_dims[decrease_axis[i]],
          1,
          phi::errors::InvalidArgument(
              "the size of decrease dimension should be 1, but received %d.",
              out_dims[decrease_axis[i]]));
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims_origin = phi::make_ddim(new_out_shape);
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc in_desc(x);
  MLUCnnlTensorDesc out_desc(
      out_dims_vector.size(), out_dims_vector.data(), ToCnnlDataType<T>());
  MLUCnnl::StridedSlice(dev_ctx,
                        starts_indices_vector.data(),
                        ends_indices_vector.data(),
                        strides_indices_vector.data(),
                        in_desc.get(),
                        GetBasePtr(&x),
                        out_desc.get(),
                        GetBasePtr(out));

  if (need_reverse) {
    phi::DenseTensor out_tmp;
    out_tmp.Resize(out_dims);
    dev_ctx.template Alloc<T>(&out_tmp);
    TensorCopy(dev_ctx, *out, false, &out_tmp);

    phi::DenseTensor reverse_axis;
    std::vector<int> reverse_axis_vector;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        reverse_axis_vector.push_back(axes[axis]);
      }
    }
    reverse_axis.Resize({static_cast<int>(reverse_axis_vector.size())});
    dev_ctx.template Alloc<int>(&reverse_axis);
    TensorFromVector(dev_ctx, reverse_axis_vector, dev_ctx, &reverse_axis);

    MLUCnnlTensorDesc input_desc(out_tmp);
    MLUCnnlTensorDesc output_desc(*out);
    MLUCnnl::Flip(dev_ctx,
                  reverse_axis_vector.data(),
                  reverse_axis_vector.size(),
                  input_desc.get(),
                  GetBasePtr(&out_tmp),
                  output_desc.get(),
                  GetBasePtr(out));
  }

  if (decrease_axis.size() > 0) {
    out->Resize(out_dims_origin);
  }
}

template <typename T, typename Context>
void StridedSliceRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const std::vector<int>& axes,
                           const phi::IntArray& starts,
                           const phi::IntArray& ends,
                           const phi::IntArray& strides,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& decrease_axis,
                           phi::DenseTensor* out) {
  int rank = x.dims().size();
  switch (rank) {
    case 1:
      custom_kernel::StridedSliceCompute<T, Context, 1>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 2:
      custom_kernel::StridedSliceCompute<T, Context, 2>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 3:
      custom_kernel::StridedSliceCompute<T, Context, 3>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 4:
      custom_kernel::StridedSliceCompute<T, Context, 4>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 5:
      custom_kernel::StridedSliceCompute<T, Context, 5>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 6:
      custom_kernel::StridedSliceCompute<T, Context, 6>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 7:
      custom_kernel::StridedSliceCompute<T, Context, 7>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 8:
      custom_kernel::StridedSliceCompute<T, Context, 8>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The rank of input is supported up to 8."));
      break;
  }
}

template <typename T, typename Context>
void BilinearKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    const phi::DenseTensor& weight,
                    const paddle::optional<phi::DenseTensor>& bias,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto batch_size = x.dims()[0];
  auto weight_dims = weight.dims();
  int out_dim = weight_dims[0];
  auto x_dim = weight_dims[1];
  auto y_dim = weight_dims[2];

  // Create the intermediate variable to calculate the result of
  // Input(X) multiplied by Input(Weight_i), the formula is:
  // left_mul = X Weight_i.
  Tensor left_mul;
  left_mul.Resize(phi::make_ddim({batch_size, y_dim}));
  dev_ctx.template Alloc<T>(&left_mul);

  MLUCnnlTensorDesc x_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc weight_desc(weight, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());
  MLUCnnlTensorDesc left_mul_desc(
      left_mul, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  phi::DenseTensor output_mat_slice;
  output_mat_slice.Resize(phi::make_ddim({batch_size}));

  phi::DenseTensor out_temp;
  out_temp.Resize(out->dims());
  dev_ctx.template Alloc<T>(&out_temp);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), &out_temp);

  for (int64_t i = 0; i < out_dim; ++i) {
    phi::DenseTensor weight_slice;
    weight_slice.Resize(phi::make_ddim({x_dim, y_dim}));
    dev_ctx.template Alloc<T>(&weight_slice);
    MLUCnnlTensorDesc weight_slice_desc(weight_slice);

    phi::DenseTensor matmul_out;
    matmul_out.Resize(phi::make_ddim({batch_size, y_dim}));
    dev_ctx.template Alloc<T>(&matmul_out);
    MLUCnnlTensorDesc matmul_out_desc(matmul_out);
    int64_t next_i = i + 1;
    int64_t value = 1;
    const phi::IntArray& starts_indices = {i};
    const phi::IntArray& ends_indices = {next_i};
    const phi::IntArray& strides_indices = {value};
    std::vector<int> infer_flags(1);
    std::vector<int> decrease_axis;
    std::vector<int> axes = {0};
    custom_kernel::StridedSliceRawKernel<T, Context>(dev_ctx,
                                                     weight,
                                                     axes,
                                                     starts_indices,
                                                     ends_indices,
                                                     strides_indices,
                                                     infer_flags,
                                                     decrease_axis,
                                                     &weight_slice);

    MLUCnnl::Matmul(dev_ctx,
                    false,
                    false,
                    x_desc.get(),
                    GetBasePtr(&x),
                    weight_slice_desc.get(),
                    GetBasePtr(&weight_slice),
                    left_mul_desc.get(),
                    GetBasePtr(&left_mul));

    int axis = -1;
    MLUOpTensorKernel<T>(
        dev_ctx, left_mul, y, axis, CNNL_OP_TENSOR_MUL, &matmul_out);

    phi::DenseTensor sum_out;
    sum_out.Resize({batch_size});
    const std::vector<int64_t>& dims = {1};
    MLUReduceOp<T>(dev_ctx,
                   matmul_out,
                   dims,
                   false,
                   /*keep_dim*/ false,
                   /*reduce_all*/ "reduce_sum",
                   &sum_out);

    std::vector<int64_t> sum_axes = {1};
    std::vector<int64_t> decrease_axes;
    std::vector<int64_t> none_axes;
    custom_kernel::SetTensorValueKernel<T, Context>(dev_ctx,
                                                    *&out_temp,
                                                    sum_out,
                                                    starts_indices,
                                                    ends_indices,
                                                    strides_indices,
                                                    sum_axes,
                                                    decrease_axes,
                                                    none_axes,
                                                    &output_mat_slice);
  }

  if (bias.get_ptr()) {
    phi::DenseTensor new_bias;
    new_bias = bias.get();
    int axis = -1;
    MLUOpTensorKernel<T>(
        dev_ctx, out_temp, new_bias, axis, CNNL_OP_TENSOR_ADD, out);
  } else {
    TensorCopy(dev_ctx, out_temp, false, out);
  }
}

template <typename T, typename Context>
void BilinearGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        const phi::DenseTensor& weight,
                        const phi::DenseTensor& dout,
                        phi::DenseTensor* dx,
                        phi::DenseTensor* dy,
                        phi::DenseTensor* dweight,
                        phi::DenseTensor* dbias) {
  auto batch_size = x.dims()[0];
  auto weight_dims = weight.dims();
  int out_dim = weight_dims[0];
  auto x_dim = weight_dims[1];
  auto y_dim = weight_dims[2];

  // Create the intermediate variable to calculate the Output(Y@Grad).
  phi::DenseTensor x_scale;
  x_scale.Resize(phi::make_ddim({batch_size, x_dim}));
  dev_ctx.template Alloc<T>(&x_scale);

  // Create the intermediate variable to calculate the Output(X@Grad).
  phi::DenseTensor y_scale;
  y_scale.Resize(phi::make_ddim({batch_size, y_dim}));
  dev_ctx.template Alloc<T>(&y_scale);

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), dx);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), dy);
  }
  if (dweight) {
    dev_ctx.template Alloc<T>(dweight);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), dweight);
  }

  if (dx || dy || dweight) {
    phi::DenseTensor dx_temp;
    dx_temp.Resize(dx->dims());
    dev_ctx.template Alloc<T>(&dx_temp);
    MLUCnnlTensorDesc dx_temp_desc(dx_temp);

    phi::DenseTensor dy_temp;
    dy_temp.Resize(dy->dims());
    dev_ctx.template Alloc<T>(&dy_temp);
    MLUCnnlTensorDesc dy_temp_desc(dy_temp);

    phi::DenseTensor dweight_temp;
    dweight_temp.Resize(phi::make_ddim({x_dim, y_dim}));
    dev_ctx.template Alloc<T>(&dweight_temp);
    MLUCnnlTensorDesc dweight_temp_desc(dweight_temp);

    for (int64_t i = 0; i < out_dim; ++i) {
      phi::DenseTensor weight_slice;
      weight_slice.Resize(phi::make_ddim({x_dim, y_dim}));
      dev_ctx.template Alloc<T>(&weight_slice);
      int64_t next_i = i + 1;
      int64_t value = 1;
      const phi::IntArray& starts_indices = {i};
      const phi::IntArray& ends_indices = {next_i};
      const phi::IntArray& strides_indices = {value};
      std::vector<int> infer_flags(1);
      std::vector<int> decrease_axis;
      std::vector<int> axes = {0};
      custom_kernel::StridedSliceRawKernel<T, Context>(dev_ctx,
                                                       weight,
                                                       axes,
                                                       starts_indices,
                                                       ends_indices,
                                                       strides_indices,
                                                       infer_flags,
                                                       decrease_axis,
                                                       &weight_slice);
      weight_slice.Resize(phi::make_ddim({x_dim, y_dim}));
      MLUCnnlTensorDesc weight_slice_desc(weight_slice);
      MLUCnnlTensorDesc x_scale_desc(x_scale);
      MLUCnnlTensorDesc y_scale_desc(y_scale);
      MLUCnnlTensorDesc dx_desc(*dx);
      MLUCnnlTensorDesc dy_desc(*dy);
      MLUCnnlTensorDesc y_desc(y);

      // dout[:, i]
      std::vector<int> dout_axes = {1};
      std::vector<int> decrease_axes;
      phi::DenseTensor dout_mat_slice;
      dout_mat_slice.Resize(phi::make_ddim({batch_size}));
      custom_kernel::StridedSliceRawKernel<T, Context>(dev_ctx,
                                                       dout,
                                                       dout_axes,
                                                       starts_indices,
                                                       ends_indices,
                                                       strides_indices,
                                                       infer_flags,
                                                       decrease_axis,
                                                       &dout_mat_slice);
      if (dx) {
        int axis = -1;
        dout_mat_slice.Resize({batch_size, 1});
        MLUCnnlTensorDesc dout_mat_slice_desc(dout_mat_slice);
        MLUOpTensorKernel<T>(
            dev_ctx, dout_mat_slice, y, axis, CNNL_OP_TENSOR_MUL, &y_scale);
        MLUCnnl::Matmul(dev_ctx,
                        false,
                        true,
                        y_scale_desc.get(),
                        GetBasePtr(&y_scale),
                        weight_slice_desc.get(),
                        GetBasePtr(&weight_slice),
                        dx_temp_desc.get(),
                        GetBasePtr(&dx_temp));
        MLUOpTensorKernel<T>(
            dev_ctx, dx_temp, *dx, axis, CNNL_OP_TENSOR_ADD, dx);
      }
      if (dy || dweight) {
        int axis = -1;
        dout_mat_slice.Resize({batch_size, 1});
        MLUCnnlTensorDesc dout_mat_slice_desc(dout_mat_slice);
        MLUOpTensorKernel<T>(
            dev_ctx, dout_mat_slice, x, axis, CNNL_OP_TENSOR_MUL, &x_scale);
        if (dy) {
          MLUCnnl::Matmul(dev_ctx,
                          false,
                          false,
                          x_scale_desc.get(),
                          GetBasePtr(&x_scale),
                          weight_slice_desc.get(),
                          GetBasePtr(&weight_slice),
                          dy_temp_desc.get(),
                          GetBasePtr(&dy_temp));
          MLUOpTensorKernel<T>(
              dev_ctx, dy_temp, *dy, axis, CNNL_OP_TENSOR_ADD, dy);
        }
        if (dweight) {
          MLUCnnl::Matmul(dev_ctx,
                          true,
                          false,
                          x_scale_desc.get(),
                          GetBasePtr(&x_scale),
                          y_desc.get(),
                          GetBasePtr(&y),
                          dweight_temp_desc.get(),
                          GetBasePtr(&dweight_temp));

          std::vector<int64_t> dweight_axes = {0};
          std::vector<int64_t> decrease_axes;
          std::vector<int64_t> none_axes;
          phi::DenseTensor dweight_slice;
          dweight_slice.Resize(phi::make_ddim({x_dim, y_dim}));
          dev_ctx.template Alloc<T>(&dweight_slice);
          MLUCnnlTensorDesc dweight_slice_desc(dweight_slice);
          custom_kernel::SetTensorValueKernel<T, Context>(dev_ctx,
                                                          *dweight,
                                                          dweight_temp,
                                                          starts_indices,
                                                          ends_indices,
                                                          strides_indices,
                                                          dweight_axes,
                                                          decrease_axes,
                                                          none_axes,
                                                          &dweight_slice);
        }
      }
    }
    // calculate the gradient of Input(Bias).
    if (dbias) {
      dev_ctx.template Alloc<T>(dbias);
      const std::vector<int64_t>& dims = {0};
      MLUReduceOp<T>(dev_ctx,
                     dout,
                     dims,
                     false, /*keep_dim*/
                     false, /*reduce_all*/
                     "reduce_sum",
                     dbias);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    bilinear, mlu, ALL_LAYOUT, custom_kernel::BilinearKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(bilinear_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearGradKernel,
                          float,
                          double) {}
