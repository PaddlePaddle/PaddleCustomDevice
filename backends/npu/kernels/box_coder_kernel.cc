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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

enum class BoxCodeType { kEncodeCenterSize = 0, kDecodeCenterSize = 1 };

BoxCodeType GetBoxCodeType(const std::string& type) {
  PADDLE_ENFORCE_EQ(
      (type == "encode_center_size") || (type == "decode_center_size"),
      true,
      phi::errors::InvalidArgument(
          "The 'code_type' attribute in BoxCoder"
          " must be 'encode_center_size' or 'decode_center_size'. "
          "But received 'code_type' is %s",
          type));
  if (type == "encode_center_size") {
    return BoxCodeType::kEncodeCenterSize;
  } else {
    return BoxCodeType::kDecodeCenterSize;
  }
}

template <typename T, typename Context>
struct BoxCoderFunction {
 public:
  explicit BoxCoderFunction(const Context& dev_ctx) : dev_ctx(dev_ctx) {
    place = dev_ctx.GetPlace();
    stream = dev_ctx.stream();
  }
  phi::DenseTensor Adds(const phi::DenseTensor& x, float scalar) {
    phi::DenseTensor y;
    y.Resize(x.dims());
    dev_ctx.template Alloc<T>(&y);
    const auto& runner = NpuOpRunner("Adds", {x}, {y}, {{"value", scalar}});
    runner.Run(stream);
    return y;
  }
  phi::DenseTensor Muls(const phi::DenseTensor& x, float scalar) {
    phi::DenseTensor y;
    y.Resize(x.dims());
    dev_ctx.template Alloc<T>(&y);
    const auto& runner = NpuOpRunner("Muls", {x}, {y}, {{"value", scalar}});
    runner.Run(stream);
    return y;
  }
  phi::DenseTensor Mul(const phi::DenseTensor& x, const phi::DenseTensor& y) {
    phi::DenseTensor z;
    z.Resize(x.dims());
    dev_ctx.template Alloc<T>(&z);
    const auto& runner = NpuOpRunner("Mul", {x, y}, {z}, {});
    runner.Run(stream);
    return z;
  }
  phi::DenseTensor SubWithBroadCast(const phi::DenseTensor& x,
                                    const phi::DenseTensor& y,
                                    const phi::DDim& shape) {
    phi::DenseTensor z;
    z.Resize(shape);
    dev_ctx.template Alloc<T>(&z);
    const auto& runner = NpuOpRunner("Sub", {x, y}, {z}, {});
    runner.Run(stream);
    return z;
  }
  void DivWithBroadCastVoid(const phi::DenseTensor& x,
                            const phi::DenseTensor& y,
                            const phi::DDim& shape,
                            phi::DenseTensor* z) {
    z->Resize(shape);
    dev_ctx.template Alloc<T>(z);
    const auto& runner = NpuOpRunner("Div", {x, y}, {*z}, {});
    runner.Run(stream);
  }
  phi::DenseTensor DivWithBroadCast(const phi::DenseTensor& x,
                                    const phi::DenseTensor& y,
                                    const phi::DDim& shape) {
    phi::DenseTensor z;
    DivWithBroadCastVoid(x, y, shape, &z);
    return z;
  }
  void MulWithBroadCastVoid(const phi::DenseTensor& x,
                            const phi::DenseTensor& y,
                            const phi::DDim& shape,
                            phi::DenseTensor* z) {
    z->Resize(shape);
    dev_ctx.template Alloc<T>(z);
    const auto& runner = NpuOpRunner("Mul", {x, y}, {*z}, {});
    runner.Run(stream);
  }
  phi::DenseTensor MulWithBroadCast(const phi::DenseTensor& x,
                                    const phi::DenseTensor& y,
                                    const phi::DDim& shape) {
    phi::DenseTensor z;
    MulWithBroadCastVoid(x, y, shape, &z);
    return z;
  }
  void AddWithBroadCastVoid(const phi::DenseTensor& x,
                            const phi::DenseTensor& y,
                            const phi::DDim& shape,
                            phi::DenseTensor* z) {
    z->Resize(shape);
    dev_ctx.template Alloc<T>(z);
    const auto& runner = NpuOpRunner("AddV2", {x, y}, {*z}, {});
    runner.Run(stream);
  }
  phi::DenseTensor AddWithBroadCast(const phi::DenseTensor& x,
                                    const phi::DenseTensor& y,
                                    const phi::DDim& shape) {
    phi::DenseTensor z;
    AddWithBroadCastVoid(x, y, shape, &z);
    return z;
  }
  phi::DenseTensor Abs(const phi::DenseTensor& x) {
    phi::DenseTensor y;
    y.Resize(x.dims());
    dev_ctx.template Alloc<T>(&y);
    const auto& runner = NpuOpRunner("Abs", {x}, {y}, {});
    runner.Run(stream);
    return y;
  }
  phi::DenseTensor Log(const phi::DenseTensor& x) {
    phi::DenseTensor t_x_m1 = Adds(x, -1);
    phi::DenseTensor y;
    y.Resize(x.dims());
    dev_ctx.template Alloc<T>(&y);
    const auto& runner = NpuOpRunner("Log1p", {t_x_m1}, {y}, {});
    runner.Run(stream);
    return y;
  }
  phi::DenseTensor Exp(const phi::DenseTensor& x) {
    phi::DenseTensor y;
    y.Resize(x.dims());
    dev_ctx.template Alloc<T>(&y);
    const auto& runner = NpuOpRunner("Exp", {x}, {y}, {});
    runner.Run(stream);
    return y;
  }
  phi::DenseTensor Dot(const phi::DenseTensor& x, const phi::DenseTensor& y) {
    auto dim_x = x.dims();
    auto dim_y = y.dims();
    PADDLE_ENFORCE_EQ(
        dim_x.size(),
        2,
        phi::errors::InvalidArgument(
            "x should be a 2-dim tensor, but got %d-dim.", dim_x.size()));
    PADDLE_ENFORCE_EQ(
        dim_y.size(),
        2,
        phi::errors::InvalidArgument(
            "y should be a 2-dim tensor, but got %d-dim.", dim_y.size()));
    PADDLE_ENFORCE_EQ(
        dim_x[1],
        dim_y[0],
        phi::errors::InvalidArgument("Expect dim_x[1] == dim_y[0], but "
                                     "got dim_x[1] = %d, dim_y[0] = %d.",
                                     dim_x[1],
                                     dim_y[0]));
    phi::DenseTensor z;
    z.Resize({dim_x[0], dim_y[1]});
    dev_ctx.template Alloc<T>(&z);
    const auto& runner =
        NpuOpRunner("MatMul",
                    {x, y},
                    {z},
                    {{"transpose_x1", false}, {"transpose_x2", false}});
    runner.Run(stream);
    return z;
  }
  void ConcatVoid(const std::vector<phi::DenseTensor>& inputs,
                  const phi::DDim& shape_out,
                  int axis,
                  phi::DenseTensor* output) {
    output->Resize(shape_out);
    dev_ctx.template Alloc<T>(output);
    std::vector<std::string> names;
    for (size_t i = 0; i < inputs.size(); i++) {
      names.push_back("x" + std::to_string(i));
    }
    NpuOpRunner runner{
        "ConcatD",
        {inputs},
        {*output},
        {{"concat_dim", axis}, {"N", static_cast<int>(inputs.size())}}};
    runner.AddInputNames(names);
    runner.Run(stream);
  }
  phi::DenseTensor Concat(const std::vector<phi::DenseTensor>& inputs,
                          const phi::DDim& shape_out,
                          int axis) {
    phi::DenseTensor output;
    ConcatVoid(inputs, shape_out, axis, &output);
    return output;
  }
  phi::DenseTensor Slice(const phi::DenseTensor& x,
                         const std::vector<int>& offsets,
                         const std::vector<int>& size,
                         const phi::DDim& shape) {
    phi::DenseTensor y;
    y.Resize(shape);
    dev_ctx.template Alloc<T>(&y);
    NpuOpRunner slice_runner;
    slice_runner.SetType("Slice")
        .AddInput(x)
        .AddInput(dev_ctx, std::move(offsets))
        .AddInput(dev_ctx, std::move(size))
        .AddOutput(y);
    slice_runner.Run(stream);
    return y;
  }

 private:
  phi::Place place;
  aclrtStream stream;
  const Context& dev_ctx;
};

template <typename T, typename Context>
void Vector2Tensor(const Context& dev_ctx,
                   const std::vector<T>& vec,
                   const phi::DDim& ddim,
                   phi::DenseTensor* tsr) {
  custom_kernel::TensorFromVector<T>(dev_ctx, vec, dev_ctx, tsr);
  tsr->Resize(ddim);
}

template <typename T, typename Context>
void BoxCoderEncCpu(const Context& dev_ctx,
                    const phi::DenseTensor* tb,
                    const phi::DenseTensor* pb,
                    const phi::DenseTensor* pbv,
                    const bool normalized,
                    const std::vector<float>& variance,
                    phi::DenseTensor* out) {
  int64_t row = tb->dims()[0];
  int64_t col = pb->dims()[0];
  int64_t len = pb->dims()[1];
  std::vector<T> target_box_data, prior_box_data, prior_box_var_data;
  std::vector<T> output(row * col * len, 0);
  custom_kernel::TensorToVector(dev_ctx, *tb, dev_ctx, &target_box_data);
  custom_kernel::TensorToVector(dev_ctx, *pb, dev_ctx, &prior_box_data);
  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      size_t offset = i * col * len + j * len;
      T prior_box_width = prior_box_data[j * len + 2] -
                          prior_box_data[j * len] + (normalized == false);
      T prior_box_height = prior_box_data[j * len + 3] -
                           prior_box_data[j * len + 1] + (normalized == false);
      T prior_box_center_x = prior_box_data[j * len] + prior_box_width / 2;
      T prior_box_center_y = prior_box_data[j * len + 1] + prior_box_height / 2;
      T target_box_center_x =
          (target_box_data[i * len + 2] + target_box_data[i * len]) / 2;
      T target_box_center_y =
          (target_box_data[i * len + 3] + target_box_data[i * len + 1]) / 2;
      T target_box_width = target_box_data[i * len + 2] -
                           target_box_data[i * len] + (normalized == false);
      T target_box_height = target_box_data[i * len + 3] -
                            target_box_data[i * len + 1] +
                            (normalized == false);
      output[offset] =
          (target_box_center_x - prior_box_center_x) / prior_box_width;
      output[offset + 1] =
          (target_box_center_y - prior_box_center_y) / prior_box_height;
      output[offset + 2] =
          std::log(std::fabs(target_box_width / prior_box_width));
      output[offset + 3] =
          std::log(std::fabs(target_box_height / prior_box_height));
    }
  }
  if (pbv) {
    custom_kernel::TensorToVector(dev_ctx, *pbv, dev_ctx, &prior_box_var_data);
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        for (int k = 0; k < 4; ++k) {
          size_t offset = i * col * len + j * len;
          int prior_var_offset = j * len;
          output[offset + k] /= prior_box_var_data[prior_var_offset + k];
        }
      }
    }
  } else if (!(variance.empty())) {
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        for (int k = 0; k < 4; ++k) {
          size_t offset = i * col * len + j * len;
          output[offset + k] /= static_cast<T>(variance[k]);
        }
      }
    }
  }
  TensorFromVector<float>(dev_ctx, output, dev_ctx, out);
}

template <typename T, typename Context>
void BoxCoderEnc(const Context& dev_ctx,
                 const phi::DenseTensor* tb,
                 const phi::DenseTensor* pb,
                 const phi::DenseTensor* pbv,
                 const bool norm,
                 const std::vector<float>& variance,
                 phi::DenseTensor* out) {
  auto M = pb->dims()[0];
  auto N = tb->dims()[0];
  auto shape_0 = phi::make_ddim({4, 2});
  phi::DenseTensor m_diff;
  phi::DenseTensor m_aver;
  std::vector<T> vec_diff = {static_cast<T>(-1),
                             static_cast<T>(0),
                             static_cast<T>(0),
                             static_cast<T>(-1),
                             static_cast<T>(1),
                             static_cast<T>(0),
                             static_cast<T>(0),
                             static_cast<T>(1)};
  std::vector<T> vec_aver = {static_cast<T>(0.5),
                             static_cast<T>(0),
                             static_cast<T>(0),
                             static_cast<T>(0.5),
                             static_cast<T>(0.5),
                             static_cast<T>(0),
                             static_cast<T>(0),
                             static_cast<T>(0.5)};
  Vector2Tensor<T, Context>(dev_ctx, vec_diff, shape_0, &m_diff);
  Vector2Tensor<T, Context>(dev_ctx, vec_aver, shape_0, &m_aver);

  BoxCoderFunction<T, Context> F(dev_ctx);
  phi::DenseTensor pb_xy = F.Adds(F.Dot(*pb, m_aver), (norm ? 0 : 0.5));
  phi::DenseTensor pb_wh = F.Adds(F.Dot(*pb, m_diff), (norm ? 0 : 1));
  phi::DenseTensor tb_xy = F.Dot(*tb, m_aver);
  phi::DenseTensor tb_wh = F.Adds(F.Dot(*tb, m_diff), (norm ? 0 : 1));

  pb_xy.Resize({1, M, 2});
  pb_wh.Resize({1, M, 2});
  tb_xy.Resize({N, 1, 2});
  tb_wh.Resize({N, 1, 2});

  auto shape_half = phi::make_ddim({N, M, 2});
  auto shape_full = phi::make_ddim({N, M, 4});

  phi::DenseTensor out_xy_0 = F.DivWithBroadCast(
      F.SubWithBroadCast(tb_xy, pb_xy, shape_half), pb_wh, shape_half);
  phi::DenseTensor out_wh_0 =
      F.Log(F.Abs(F.DivWithBroadCast(tb_wh, pb_wh, shape_half)));
  phi::DenseTensor out_0 = F.Concat({out_xy_0, out_wh_0}, shape_full, 2);

  if (pbv) {
    F.DivWithBroadCastVoid(out_0, *pbv, shape_full, out);
  } else {
    phi::DenseTensor t_var;
    std::vector<T> vec_var(4);
    for (auto i = 0; i < 4; i++) {
      vec_var[i] = static_cast<T>(variance[i]);
    }
    Vector2Tensor<T, Context>(
        dev_ctx, vec_var, phi::make_ddim({1, 1, 4}), &t_var);
    F.DivWithBroadCastVoid(out_0, t_var, shape_full, out);
  }
}

template <typename T, typename Context>
void BoxCoderDec(const Context& dev_ctx,
                 const phi::DenseTensor* tb,
                 const phi::DenseTensor* pb,
                 const phi::DenseTensor* pbv,
                 const bool norm,
                 const std::vector<float>& variance,
                 int axis,
                 phi::DenseTensor* out) {
  auto shape_0 = phi::make_ddim({4, 2});
  phi::DenseTensor m_diff;
  phi::DenseTensor m_aver;
  std::vector<T> vec_diff = {static_cast<T>(-1),
                             static_cast<T>(0),
                             static_cast<T>(0),
                             static_cast<T>(-1),
                             static_cast<T>(1),
                             static_cast<T>(0),
                             static_cast<T>(0),
                             static_cast<T>(1)};
  std::vector<T> vec_aver = {static_cast<T>(0.5),
                             static_cast<T>(0),
                             static_cast<T>(0),
                             static_cast<T>(0.5),
                             static_cast<T>(0.5),
                             static_cast<T>(0),
                             static_cast<T>(0),
                             static_cast<T>(0.5)};
  Vector2Tensor<T, Context>(dev_ctx, vec_diff, shape_0, &m_diff);
  Vector2Tensor<T, Context>(dev_ctx, vec_aver, shape_0, &m_aver);

  BoxCoderFunction<T, Context> F(dev_ctx);
  phi::DenseTensor pb_xy = F.Adds(F.Dot(*pb, m_aver), (norm ? 0 : 0.5));
  phi::DenseTensor pb_wh = F.Adds(F.Dot(*pb, m_diff), (norm ? 0 : 1));
  auto pb_resize_shape = axis == 0 ? phi::make_ddim({1, pb->dims()[0], 2})
                                   : phi::make_ddim({pb->dims()[0], 1, 2});
  pb_xy.Resize(pb_resize_shape);
  pb_wh.Resize(pb_resize_shape);

  auto tbox_slice_shape = phi::make_ddim({tb->dims()[0], tb->dims()[1], 2});
  std::vector<int> tbox_slice_size = {
      static_cast<int>(tb->dims()[0]), static_cast<int>(tb->dims()[1]), 2};
  phi::DenseTensor tbox01 =
      F.Slice(*tb, {0, 0, 0}, tbox_slice_size, tbox_slice_shape);
  phi::DenseTensor tbox23 =
      F.Slice(*tb, {0, 0, 2}, tbox_slice_size, tbox_slice_shape);

  phi::DenseTensor tb_xy;
  phi::DenseTensor tb_wh;
  if (pbv) {
    auto pbvt_slice_shape = phi::make_ddim({pbv->dims()[0], 2});
    auto pbvt_resize_shape = axis == 0 ? phi::make_ddim({1, pbv->dims()[0], 2})
                                       : phi::make_ddim({pbv->dims()[0], 1, 2});
    std::vector<int> pbvt_slice_size = {static_cast<int>(pbv->dims()[0]), 2};
    phi::DenseTensor pbv_t01 =
        F.Slice(*pbv, {0, 0}, pbvt_slice_size, pbvt_slice_shape);
    phi::DenseTensor pbv_t23 =
        F.Slice(*pbv, {0, 2}, pbvt_slice_size, pbvt_slice_shape);
    pbv_t01.Resize(pbvt_resize_shape);
    pbv_t23.Resize(pbvt_resize_shape);

    F.AddWithBroadCastVoid(
        F.MulWithBroadCast(tbox01, F.Mul(pb_wh, pbv_t01), tbox_slice_shape),
        pb_xy,
        tbox_slice_shape,
        &tb_xy);
    F.MulWithBroadCastVoid(
        F.Exp(F.MulWithBroadCast(pbv_t23, tbox23, tbox_slice_shape)),
        pb_wh,
        tbox_slice_shape,
        &tb_wh);
  } else if (variance.empty()) {
    F.AddWithBroadCastVoid(F.MulWithBroadCast(tbox01, pb_wh, tbox_slice_shape),
                           pb_xy,
                           tbox_slice_shape,
                           &tb_xy);
    F.MulWithBroadCastVoid(F.Exp(tbox23), pb_wh, tbox_slice_shape, &tb_wh);
  } else {
    phi::DenseTensor t_var01, t_var23;
    auto t_var_shape = phi::make_ddim({1, 1, 2});
    std::vector<T> vec_var01 = {static_cast<T>(variance[0]),
                                static_cast<T>(variance[1])};
    std::vector<T> vec_var23 = {static_cast<T>(variance[2]),
                                static_cast<T>(variance[3])};
    Vector2Tensor<T, Context>(dev_ctx, vec_var01, t_var_shape, &t_var01);
    Vector2Tensor<T, Context>(dev_ctx, vec_var23, t_var_shape, &t_var23);
    F.AddWithBroadCastVoid(
        F.MulWithBroadCast(tbox01,
                           F.MulWithBroadCast(pb_wh, t_var01, pb_resize_shape),
                           tbox_slice_shape),
        pb_xy,
        tbox_slice_shape,
        &tb_xy);
    F.MulWithBroadCastVoid(
        F.Exp(F.MulWithBroadCast(t_var23, tbox23, tbox_slice_shape)),
        pb_wh,
        tbox_slice_shape,
        &tb_wh);
  }
  phi::DenseTensor obox01 =
      F.AddWithBroadCast(tb_xy, F.Muls(tb_wh, -0.5), tbox_slice_shape);
  phi::DenseTensor obox23 =
      F.Adds(F.AddWithBroadCast(tb_xy, F.Muls(tb_wh, 0.5), tbox_slice_shape),
             (norm ? 0 : -1));
  F.ConcatVoid({obox01, obox23}, out->dims(), 2, out);
}

template <typename T, typename Context>
void BoxCoderKernel(const Context& dev_ctx,
                    const phi::DenseTensor& prior_box,
                    const paddle::optional<phi::DenseTensor>& prior_box_var,
                    const phi::DenseTensor& target_box,
                    const std::string& code_type,
                    bool box_normalized,
                    int axis,
                    const std::vector<float>& variance,
                    phi::DenseTensor* output_box) {
  if (target_box.lod().size()) {
    PADDLE_ENFORCE_EQ(target_box.lod().size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "Input(TargetBox) of BoxCoder operator "
                          "supports LoD with only one level. But received "
                          "level = %d",
                          target_box.lod().size()));
  }
  if (prior_box_var) {
    PADDLE_ENFORCE_EQ(variance.empty(),
                      true,
                      phi::errors::InvalidArgument(
                          "Input 'PriorBoxVar' and attribute 'variance' "
                          "of BoxCoder operator should not be used at the "
                          "same time."));
  }
  if (!(variance.empty())) {
    PADDLE_ENFORCE_EQ(
        static_cast<int>(variance.size()),
        4,
        phi::errors::InvalidArgument("Size of attribute 'variance' of BoxCoder "
                                     "operator should be 4. But received "
                                     "size = %d",
                                     variance.size()));
  }
  auto code_type_data = GetBoxCodeType(code_type);
  if (code_type_data == BoxCodeType::kEncodeCenterSize) {
    if (target_box.dtype() == phi::DataType::FLOAT32) {
      // TODO(duanyanhui): Ascend op "MatMul" transform the fp32
      // input to fp16, which will bring diff to the following
      // calculation. In this kernel, the diff in "MatMul" op will
      // be expanded in the following "Div" op.
      // For example, 1e-4 diff is generated in "MatMul" op, and
      // the expect value is 1e-4 and the actual value is 2e-4.
      // Assuming the form of "Div" op is a/b, and a equals to 1e-2,
      // then the diff will be expanded to 1e+2. So, We use cpu to
      // do the calculation.
      auto row = target_box.dims()[0];
      auto col = prior_box.dims()[0];
      auto len = prior_box.dims()[1];
      output_box->Resize({1, row * col * len});
      dev_ctx.template Alloc<T>(output_box);
      BoxCoderEncCpu<float, Context>(dev_ctx,
                                     &target_box,
                                     &prior_box,
                                     prior_box_var.get_ptr(),
                                     box_normalized,
                                     variance,
                                     output_box);
      output_box->Resize({row, col, len});
    } else {
      BoxCoderEnc<T, Context>(dev_ctx,
                              &target_box,
                              &prior_box,
                              prior_box_var.get_ptr(),
                              box_normalized,
                              variance,
                              output_box);
    }
  } else {
    BoxCoderDec<T, Context>(dev_ctx,
                            &target_box,
                            &prior_box,
                            prior_box_var.get_ptr(),
                            box_normalized,
                            variance,
                            axis,
                            output_box);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(box_coder,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BoxCoderKernel,
                          float,
                          phi::dtype::float16) {}
