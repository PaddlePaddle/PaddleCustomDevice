
#include <acl/acl.h>
#include "atb_layer_base.h"

#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAscendAtbOpBase> g_atbsetvalue;

std::vector<int64_t> get_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (auto i = 0; i < strides.size() - 1; ++i) {
        for (auto j = i + 1; j < strides.size(); ++j) {
            strides[i] *= shape[j];
        }
    }
    return strides;
}

void AtbSetValueOp(const paddle::Tensor& x,
                   const paddle::Tensor& value,
                   const std::vector<int64_t>& starts,
                   const std::vector<int64_t>& ends) {

    // std::stringstream ss;
    // ss << "set_value: x.shape=" << phi::make_ddim(x.shape())
    //     << ", value.shape=" << phi::make_ddim(value.shape());
    // ss << ", indices=";
    // for (auto i = 0; i < starts.size(); ++i) {
    //     ss << starts[i] << ":" << ends[i];
    //     if (i < starts.size() - 1) {
    //         ss << ", ";
    //     }
    // }
    // LOG(INFO) << ss.str();

    // infer shape & dtype
    auto place = x.place();
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(place));
    auto stream = static_cast<aclrtStream>(dev_ctx->stream());
    auto x_dtype = x.dtype();
    auto x_shape = x.shape();
    PADDLE_ENFORCE(
        x_shape.size() == starts.size() && x_shape.size() == ends.size(),
        phi::errors::InvalidArgument(
            "x.shape().size() must be equal to starts.size() and ends.size()"));

    std::vector<int64_t> starts_tmp = starts;
    std::vector<int64_t> ends_tmp = ends;
    std::vector<int64_t> size_tmp(x_shape.size(), 0);
    for (auto i = 0; i < x_shape.size(); ++i) {
        starts_tmp[i] =
            starts_tmp[i] < 0 ? starts_tmp[i] + x_shape[i] : starts_tmp[i];
        ends_tmp[i] = ends_tmp[i] <= 0 ? ends_tmp[i] + x_shape[i] : ends_tmp[i];
        size_tmp[i] = ends_tmp[i] - starts_tmp[i];
    }

    int the_last_diff_axis = 0;
    int the_last_second_diff_axis = 0;
    int diff_axis_num = 0;
    for (auto i = 0; i < x_shape.size(); ++i) {
        if (size_tmp[i] != x_shape[i]) {
            diff_axis_num++;
            the_last_second_diff_axis =
                diff_axis_num > 1 ? the_last_diff_axis : 0;
            the_last_diff_axis = i;
        }
    }

    auto max_dim = x_shape[0];
    for (auto i = 1; i < x_shape.size(); ++i) {
        max_dim = std::max(max_dim, x_shape[i]);
    }

    if (diff_axis_num == 0) {
        C_Device_st device{dev_ctx->GetPlace().GetDeviceId()};
        C_Stream stream = reinterpret_cast<C_Stream>(dev_ctx->stream());
        AsyncMemCpyD2D(&device,
                       stream,
                       const_cast<void*>(x.data()),
                       const_cast<void*>(value.data()),
                       value.numel() * phi::SizeOf(x_dtype));
    } else if (diff_axis_num == 1) {
        if (the_last_diff_axis == 0) {
            uint64_t offset = starts[0] *
                              std::accumulate(x_shape.begin() + 1,
                                              x_shape.end(),
                                              1,
                                              std::multiplies<int64_t>()) *
                              phi::SizeOf(x_dtype);
            C_Device_st device{dev_ctx->GetPlace().GetDeviceId()};
            C_Stream stream = reinterpret_cast<C_Stream>(dev_ctx->stream());
            AsyncMemCpyD2D(&device,
                           stream,
                           reinterpret_cast<void*>(
                               reinterpret_cast<uint64_t>(x.data()) + offset),
                           const_cast<void*>(value.data()),
                           value.numel() * phi::SizeOf(x_dtype));
        } else {
            atb::Operation *op = nullptr;
            atb::infer::SetValueParam param;
            param.starts.resize(x_shape.size());
            param.ends.resize(x_shape.size());
            param.strides.resize(x_shape.size());
            for (auto i = 0; i < x_shape.size(); ++i) {
                param.starts[i] = starts_tmp[i];
                param.ends[i] = ends_tmp[i];
                param.strides[i] = 1;
            }
            g_atbsetvalue.reset(new PpAscendAtbOpBase("AtbSetValue"));
            atb::CreateOperation(param, &op);
            g_atbsetvalue->operation_.reset(op);

            std::vector<const phi::DenseTensor *> inputs;
            inputs.push_back(static_cast<const phi::DenseTensor *>(x.impl().get()));
            inputs.push_back(static_cast<const phi::DenseTensor *>(value.impl().get()));

            std::vector<const phi::DenseTensor *> outputs;
            outputs.resize(0);

            g_atbsetvalue->Execute(stream, inputs, outputs);
        }
    } else {
        if (diff_axis_num == 2 &&
            (x_shape[the_last_diff_axis] == max_dim ||
             x_shape[the_last_second_diff_axis] == max_dim)) {
            atb::Operation *op = nullptr;
            atb::infer::SetValueParam param;
            param.starts.resize(x_shape.size());
            param.ends.resize(x_shape.size());
            param.strides.resize(x_shape.size());
            for (auto i = 0; i < x_shape.size(); ++i) {
                param.starts[i] = starts_tmp[i];
                param.ends[i] = ends_tmp[i];
                param.strides[i] = 1;
            }
            g_atbsetvalue.reset(new PpAscendAtbOpBase("AtbSetValue"));
            atb::CreateOperation(param, &op);
            g_atbsetvalue->operation_.reset(op);

            std::vector<const phi::DenseTensor *> inputs;
            inputs.push_back(static_cast<const phi::DenseTensor *>(x.impl().get()));
            inputs.push_back(static_cast<const phi::DenseTensor *>(value.impl().get()));

            std::vector<const phi::DenseTensor *> outputs;
            outputs.resize(0);

            g_atbsetvalue->Execute(stream, inputs, outputs);
        } else {
            atb::Operation *op = nullptr;
            atb::infer::SetValueParam param;
            param.starts.resize(x_shape.size());
            param.ends.resize(x_shape.size());
            param.strides.resize(x_shape.size());

            std::vector<int64_t> x_stride = get_strides(x_shape);
            std::vector<int64_t> value_stride = get_strides(size_tmp);
            std::vector<int64_t> offset_stride =
                get_strides(std::vector<int64_t>(
                    size_tmp.begin(),
                    size_tmp.begin() + the_last_second_diff_axis + 1));

            uint64_t index_end = std::accumulate(
                size_tmp.begin(),
                size_tmp.begin() + the_last_second_diff_axis + 1,
                1,
                std::multiplies<int64_t>());
            auto x_shape_tmp = x_shape;
            auto value_shape = size_tmp;
            for (auto i = 0; i < the_last_second_diff_axis + 1; ++i) {
                x_shape_tmp[i] = 1;
                value_shape[i] = 1;
            }
            for (uint64_t index = 0; index < index_end; ++index) {
                for (auto i = 0; i < x_shape.size(); ++i) {
                    if (i < the_last_second_diff_axis + 1) {
                        param.starts[i] = 0;
                        param.ends[i] = 1;
                    } else {
                        param.starts[i] = starts_tmp[i];
                        param.ends[i] = ends_tmp[i];
                    }
                    param.strides[i] = 1;
                }
                uint64_t x_data_offset = 0;
                uint64_t value_data_offset = 0;
                for (auto i = 0; i < the_last_second_diff_axis + 1; ++i) {
                    uint64_t offset = 0;
                    if (i == 0) {
                        offset = index / offset_stride[i];
                    } else {
                        offset =
                            (index % offset_stride[i - 1]) / offset_stride[i];
                    }
                    x_data_offset += (starts_tmp[i] + offset) * x_stride[i] *
                                     phi::SizeOf(x_dtype);
                    value_data_offset +=
                        offset * value_stride[i] * phi::SizeOf(x_dtype);
                }

                auto* x_data = reinterpret_cast<void*>(
                    reinterpret_cast<uint64_t>(x.data()) + x_data_offset);
                auto* value_data = reinterpret_cast<void*>(
                    reinterpret_cast<uint64_t>(value.data()) +
                    value_data_offset);
            g_atbsetvalue.reset(new PpAscendAtbOpBase("AtbSetValue"));
            atb::CreateOperation(param, &op);
            g_atbsetvalue->operation_.reset(op);

            std::vector<const phi::DenseTensor *> inputs;
            inputs.push_back(static_cast<const phi::DenseTensor *>(x.impl().get()));
            inputs.push_back(static_cast<const phi::DenseTensor *>(value.impl().get()));

            std::vector<const phi::DenseTensor *> outputs;
            outputs.resize(0);

            g_atbsetvalue->Execute(stream, inputs, outputs);
            }
        }
    }
}

PD_BUILD_OP(atb_set_value)
    .Inputs({"x", "value"})
    .Outputs({"out"})
    .SetInplaceMap({{"x", "out"}})
    .Attrs({"starts: std::vector<int64_t>", "ends: std::vector<int64_t>"})
    .SetKernelFn(PD_KERNEL(AtbSetValueOp));
