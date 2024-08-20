// // Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// #include "funcs.h"
// #include "hpu_operator.h"
// #include "perf_lib_layer_params.h"

// namespace custom_kernel {
// class Full : public HpuOperator {
//  public:
//   Full() : HpuOperator("full_fwd_f32") {}
//   template <typename VType>
//   void AddNode(const std::vector<DIMS>& ins,
//                const std::vector<DIMS>& outs,
//                VType val) {
//     assert(ins.size() == 0 && "input size should be 0");
//     assert(outs.size() == 1 && "output size should be 1");

//     synTensor outputs[outs.size()] = {
//         createTensor(outs[0].size(), syn_type_float, outs[0], true, "output")};
//     ns_ConstantKernel::Params params{val};

//     synStatus status = synNodeCreate(graphHandle_,
//                                      nullptr,
//                                      outputs,
//                                      0,
//                                      outs.size(),
//                                      &params,
//                                      sizeof(params),
//                                      "constant_f32",
//                                      "constant_fwd_op",
//                                      nullptr,
//                                      nullptr);
//     CHKSTATUS("synNodeCreate full failed!");
//   }
// };

// template <typename T, typename VType>
// void FullValue(const phi::Context& dev_ctx,
//                phi::DenseTensor* tensor,
//                VType val) {
//   if (tensor->dims().size() == 0) tensor->Resize({1});
//   auto t = dev_ctx.template Alloc<T>(tensor);

//   FullOperator op;

//   op.AddNode({}, {tensor->dims()}, val);

//   std::map<std::string, uint64_t> tensors;
//   tensors["output"] = reinterpret_cast<uint64_t>(tensor->data<T>());
//   op.CompileAndExecute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
// }

// template <typename T, typename Context>
// void FullKernel(const Context& dev_ctx,
//                 const phi::IntArray& shape,
//                 const phi::Scalar& val,
//                 phi::DataType dtype,
//                 phi::DenseTensor* out) {
//   //   auto int_shape = shape.GetData();
//   //   out->Resize(std::vector<int64_t>(int_shape.cbegin(), int_shape.cend()));
//   //   FullValue<T>(dev_ctx, out, val.to<T>());
// }
// }  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(full,
//                           intel_hpu,
//                           ALL_LAYOUT,
//                           custom_kernel::FullKernel,
//                           float,
//                           uint8_t,
//                           int16_t,
//                           int32_t,
//                           int64_t,
//                           bool) {}
