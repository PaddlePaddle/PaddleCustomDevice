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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void SetValueKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    const phi::IntArray& steps,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& decrease_axes,
                    const std::vector<int64_t>& none_axes,
                    const std::vector<int64_t>& shape,
                    const std::vector<phi::Scalar>& values,
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

  for (int i = 0; i < in_dims.size(); ++i) {
    starts_indices[i] = 0;
    ends_indices[i] = static_cast<int>(slice_dims[i]);
    strides_indices[i] = 1;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int axis_index = axes[i];
    starts_indices[axis_index] = static_cast<int64_t>(starts_local[i]);
    ends_indices[axis_index] = static_cast<int64_t>(ends_local[i]);
    strides_indices[axis_index] = static_cast<int64_t>(steps_local[i]);
  }

  std::vector<T> assgin_values;
  assgin_values.reserve(values.size());
  for (const auto& val : values) {
    assgin_values.push_back(val.to<T>());
  }
  phi::DenseTensor value_t;
  value_t.Resize(phi::make_ddim(shape));
  custom_kernel::TensorFromVector(dev_ctx, assgin_values, dev_ctx, &value_t);
  dev_ctx.Wait();
  value_t.Resize(phi::make_ddim(shape));

  phi::DenseTensor value_temp;
  if (slice_dims_for_assign == value_t.dims()) {
    value_temp = value_t;
  } else {
    value_temp.Resize(slice_dims_for_assign);
    dev_ctx.template Alloc<T>(&value_temp);
    MLUCnnlTensorDesc value_t_desc(value_t);
    MLUCnnlTensorDesc value_temp_desc(value_temp);
    MLUCnnl::BroadcastTo(dev_ctx,
                         value_t_desc.get(),
                         GetBasePtr(&value_t),
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
  for (int i = 0; i < in_dims.size(); ++i) {
    if (starts_indices[i] < 0 || ends_indices[i] < 0) {
      starts_indices[i] -= in_dims[i];
      ends_indices[i] -= in_dims[i];
    }
    if (strides_indices[i] > 0)
      index_dims[i] =
          static_cast<int>((ends_indices[i] - starts_indices[i] - 1) /
                           strides_indices[i]) +
          1;
    else
      index_dims[i] =
          static_cast<int>((ends_indices[i] - starts_indices[i] + 1) /
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
  MLUCnnlTensorDesc indices_desc(index_temp);
  MLUCnnlTensorDesc indices_out_desc(index_out);
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

  // when input x and input value's dtype is int64,
  // cast datadtype to int32 for cnnlScatterRef usage
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
}

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

  for (int i = 0; i < in_dims.size(); ++i) {
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
  for (int i = 0; i < in_dims.size(); ++i) {
    if (starts_indices[i] < 0 || ends_indices[i] < 0) {
      starts_indices[i] -= in_dims[i];
      ends_indices[i] -= in_dims[i];
    }
    if (strides_indices[i] > 0)
      index_dims[i] =
          static_cast<int>((ends_indices[i] - starts_indices[i] - 1) /
                           strides_indices[i]) +
          1;
    else
      index_dims[i] =
          static_cast<int>((ends_indices[i] - starts_indices[i] + 1) /
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

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(set_value,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SetValueKernel,
                          float,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(set_value_with_tensor,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SetTensorValueKernel,
                          float,
                          int,
                          bool,
                          int64_t) {}
