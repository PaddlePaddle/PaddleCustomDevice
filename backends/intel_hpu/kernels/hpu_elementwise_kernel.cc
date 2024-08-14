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
#include "funcs.h"
#include "glog/logging.h"
#include "hpu_operator.h"
#include "utils/utills.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "synapse_common_types.h"

namespace custom_kernel {

class BinaryOperator : public HpuOperator {
 public:
  BinaryOperator(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}
      
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               synDataType datatype) {
    assert(ins.size() == 2 && "input size should be 2");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), datatype, ins[0], true, "x"),
        createTensor(ins[1].size(), datatype, ins[1], true, "y")};
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), datatype, outs[0], true, "output")};
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     pName_.c_str(),
                                     nullptr,
                                     nullptr);
    CHKSTATUS("synNodeCreate add_fwd failed!");
  }
  std::string pName_;
};


#define BINARY_RAW_KERNEL(kernel_func, guid_prefix, node_name)             \
template <typename T, typename Context>                                    \
void kernel_func(const Context& dev_ctx,                                   \
                  const phi::DenseTensor& x,                               \
                  const phi::DenseTensor& y,                               \
                  int axis,                                                \
                  phi::DenseTensor* out) {                                 \
  dev_ctx.template Alloc<T>(out);                                          \
                                                                           \
  std::vector<int64_t> x_dim = phi::vectorize<int64_t>(x.dims());          \
  std::vector<int64_t> y_dim = phi::vectorize<int64_t>(y.dims());          \
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims()); \
                                                                           \
  OpCacheOperator op_info;                                                 \
  op_info.prepareOpInfo<T, nullptr_t>(guid_prefix, {x_dim, y_dim}, nullptr);    \
  BinaryOperator* op = static_cast<BinaryOperator*>(op_info.getOp());      \
                                                                           \
  if(op == nullptr){                                                       \
    BinaryOperator* op_new = new BinaryOperator(op_info.guid_, node_name); \
    op_new->AddNode({x_dim, y_dim}, {outputs_dim}, op_info.datatype_);     \
    op_new->Compile();                                                     \
    op = op_new;                                                           \
    op_info.setOp(op);                                                     \
  }                                                                        \
                                                                           \
  std::map<std::string, uint64_t> tensors;                                 \
  tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());                  \
  tensors["y"] = reinterpret_cast<uint64_t>(y.data<T>());                  \
  tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());          \
                                                                           \
  op->Execute(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);      \
                                                                           \
  return;                                                                  \
}                                                                          \

#define BINARY_KERNEL(kernel_func, kernel_raw_func)                        \
template <typename T, typename Context>                                    \
void kernel_func(const Context& dev_ctx,                                   \
               const phi::DenseTensor& x,                                  \
               const phi::DenseTensor& y,                                  \
               phi::DenseTensor* out) {                                    \
  int axis = -1;                                                           \
  custom_kernel::kernel_raw_func<T>(dev_ctx, x, y, axis, out);             \
}                                                                          \


//#define PRINT_MACRO_HELPER(args...) #args
//#define PRINT_MACRO(x) #x "=" PRINT_MACRO_HELPER(x)
//#pragma message(PRINT_MACRO(BINARY_KERNEL(AddRawKernel, "add_fwd", "Add")))


BINARY_RAW_KERNEL(AddRawKernel, "add_fwd", "Add")
BINARY_KERNEL(AddKernel, AddRawKernel)

}  // namespace custom_kernel

#define HPU_KERNEL_REGISTER(kernel_name, kernel_func, ...)    \
PD_REGISTER_PLUGIN_KERNEL(kernel_name,                        \
                          intel_hpu,                          \
                          ALL_LAYOUT,                         \
                          custom_kernel::kernel_func,         \
                          __VA_ARGS__) {}                     \

HPU_KERNEL_REGISTER(add_raw, AddRawKernel, float, phi::dtype::float16, phi::dtype::bfloat16)
HPU_KERNEL_REGISTER(add, AddKernel, float, phi::dtype::float16, phi::dtype::bfloat16)

