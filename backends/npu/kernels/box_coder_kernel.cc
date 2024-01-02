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
std::vector<phi::DenseTensor> GetSplitTensor(const Context& dev_ctx,
                                             const phi::DenseTensor& x,
                                             const phi::DDim shape) {
  std::vector<phi::DenseTensor> outputs;
  for (size_t i = 0; i < 4; ++i) {
    phi::DenseTensor tmp_out;
    tmp_out.Resize(shape);
    dev_ctx.template Alloc<T>(&tmp_out);
    outputs.push_back(tmp_out);
  }
  NpuOpRunner runner;
  runner.SetType("Split")
      .AddInput(dev_ctx, std::vector<int32_t>({1}))
      .AddInput(x)
      .AddOutputs(outputs)
      .AddAttrs({{"num_split", static_cast<int32_t>(4)}});
  auto stream = dev_ctx.stream();
  runner.Run(stream);
  return outputs;
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

  BoxCoderFunction<T, Context> F(dev_ctx);
  auto shape_1 = phi::make_ddim({1, M});
  auto shape_2 = phi::make_ddim({N, 1});

  std::vector<phi::DenseTensor> pb_split_tensors =
      GetSplitTensor<T, Context>(dev_ctx, *pb, shape_1);
  std::vector<phi::DenseTensor> tb_split_tensors =
      GetSplitTensor<T, Context>(dev_ctx, *tb, shape_2);
  phi::DenseTensor pb_w = F.Adds(
      F.SubWithBroadCast(pb_split_tensors[2], pb_split_tensors[0], shape_1),
      (norm ? 0 : 1));
  phi::DenseTensor pb_h = F.Adds(
      F.SubWithBroadCast(pb_split_tensors[3], pb_split_tensors[1], shape_1),
      (norm ? 0 : 1));
  phi::DenseTensor pb_x =
      F.AddWithBroadCast(pb_split_tensors[0], F.Muls(pb_w, 0.5), shape_1);
  phi::DenseTensor pb_y =
      F.AddWithBroadCast(pb_split_tensors[1], F.Muls(pb_h, 0.5), shape_1);
  phi::DenseTensor tb_x = F.AddWithBroadCast(F.Muls(tb_split_tensors[0], 0.5),
                                             F.Muls(tb_split_tensors[2], 0.5),
                                             shape_2);
  phi::DenseTensor tb_y = F.AddWithBroadCast(F.Muls(tb_split_tensors[1], 0.5),
                                             F.Muls(tb_split_tensors[3], 0.5),
                                             shape_2);
  phi::DenseTensor tb_w = F.Adds(
      F.SubWithBroadCast(tb_split_tensors[2], tb_split_tensors[0], shape_2),
      (norm ? 0 : 1));
  phi::DenseTensor tb_h = F.Adds(
      F.SubWithBroadCast(tb_split_tensors[3], tb_split_tensors[1], shape_2),
      (norm ? 0 : 1));

  auto shape_3 = phi::make_ddim({N, M});
  auto shape_full = phi::make_ddim({N, M, 4});
  phi::DenseTensor out_0 = F.DivWithBroadCast(
      F.SubWithBroadCast(tb_x, pb_x, shape_3), pb_w, shape_3);
  phi::DenseTensor out_1 = F.DivWithBroadCast(
      F.SubWithBroadCast(tb_y, pb_y, shape_3), pb_h, shape_3);
  phi::DenseTensor out_2 =
      F.Log(F.Abs(F.DivWithBroadCast(tb_w, pb_w, shape_3)));
  phi::DenseTensor out_3 =
      F.Log(F.Abs(F.DivWithBroadCast(tb_h, pb_h, shape_3)));

  out_0.Resize({N, M, 1});
  out_1.Resize({N, M, 1});
  out_2.Resize({N, M, 1});
  out_3.Resize({N, M, 1});
  phi::DenseTensor out_tmp;
  std::vector<phi::DenseTensor> out_vector = {out_0, out_1, out_2, out_3};
  F.ConcatVoid(out_vector, shape_full, 2, &out_tmp);
  if (pbv) {
    F.DivWithBroadCastVoid(out_tmp, *pbv, shape_full, out);
  } else {
    phi::DenseTensor t_var;
    std::vector<T> vec_var(4);
    for (auto i = 0; i < 4; i++) {
      vec_var[i] = static_cast<T>(variance[i]);
    }
    Vector2Tensor<T, Context>(
        dev_ctx, vec_var, phi::make_ddim({1, 1, 4}), &t_var);
    F.DivWithBroadCastVoid(out_tmp, t_var, shape_full, out);
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
    BoxCoderEnc<T, Context>(dev_ctx,
                            &target_box,
                            &prior_box,
                            prior_box_var.get_ptr(),
                            box_normalized,
                            variance,
                            output_box);
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
