#ifndef BR_PADDLE_SUPA_H_
#define BR_PADDLE_SUPA_H_

#include <map>
#include <sstream>

#include "paddle/phi/extension.h"
#include "sucl/br_cl.h"

using std::shared_ptr;

#define __INTERNAL_HANDLE_THE_ERROR try {
#define __INTERNAL_END_HANDLE_THE_ERROR                                      \
  }                                                                          \
  catch (Status & e) {                                                       \
    switch (e.Code()) {                                                      \
      case Status::kBr_InvalidArgument:                                      \
        throw ::phi::enforce::EnforceNotMet(                                 \
            ::phi::ErrorSummary(::phi::ErrorCode::INVALID_ARGUMENT,          \
                                string(e.what())),                           \
            e.File(), e.Line());                                             \
      case Status::kBr_SupaKernelError_UnImplemented:                        \
        throw ::phi::enforce::EnforceNotMet(                                 \
            ::phi::ErrorSummary(::phi::ErrorCode::UNIMPLEMENTED,             \
                                string(e.what())),                           \
            e.File(), e.Line());                                             \
      case Status::kBr_SupaKernelError_LaunchError:                          \
      case Status::kBr_Error:                                                \
        throw ::phi::enforce::EnforceNotMet(                                 \
            ::phi::ErrorSummary(::phi::ErrorCode::FATAL, string(e.what())),  \
            e.File(), e.Line());                                             \
      default:                                                               \
        throw ::phi::enforce::EnforceNotMet(                                 \
            ::phi::ErrorSummary(::phi::ErrorCode::LEGACY, string(e.what())), \
            e.File(), e.Line());                                             \
    }                                                                        \
  }

namespace br_device {

template <typename T, typename Context>
class SupaOpRunner {
 public:
  SupaOpRunner(const Context &dev_ctx, const OpParams &op_params,
               const vector<const phi::DenseTensor *> &ins,
               const vector<phi::DenseTensor *> &outs) {
    __INTERNAL_HANDLE_THE_ERROR

    op_executor_ = shared_ptr<sucl::OpExecutor>(new sucl::OpExecutor);

    for (auto itm : ins) {
      op_executor_->AddInput(GetBrTensor(itm, false));
    }
    for (auto itm : outs) {
      dev_ctx.template Alloc<T>(itm);
      op_executor_->AddOutput(GetBrTensor(itm, true));
    }

    op_executor_->CompileOp(reinterpret_cast<sucl::PStream>(dev_ctx.stream()),
                            op_params);

    __INTERNAL_END_HANDLE_THE_ERROR
  }

  void Run() {
    __INTERNAL_HANDLE_THE_ERROR
    op_executor_->Execute();
    __INTERNAL_END_HANDLE_THE_ERROR
  }

 private:
  shared_ptr<BrTensor> GetBrTensor(const phi::DenseTensor *dtensor,
                                   bool out_if) {
    if (dtensor == nullptr || dtensor->numel() == 0) {
      return shared_ptr<BrTensor>(nullptr);
    } else {
      auto shape = phi::vectorize<int64_t>(dtensor->dims());
      Dim dims(shape.empty() ? vector<int64_t>(1, 1) : shape);  // :squeeze(x)
      TensorType tensor_type(out_if ? TensorType::kOutput : TensorType::kInput);
      phi::DataType paddle_dtype = dtensor->dtype();
      DataType framework_dtype(DataTypeMapping(paddle_dtype));
      Layout layout(LayoutConvert(dtensor->layout()));

      void *data_buf = (void *)dtensor->data();

      auto btensor = sucl::OpExecutor::CreateBrTensor(
          tensor_type, dims, layout, framework_dtype, data_buf);
      return btensor;
    }
  }

  DataType DataTypeMapping(phi::DataType dtype) {
    switch (dtype) {
      case phi::DataType::FLOAT32:
        return DataType::kBr_DataType_FP32;
      case phi::DataType::UINT8:
        return DataType::kBr_DataType_U8;
      case phi::DataType::INT8:
        return DataType::kBr_DataType_S8;
      case phi::DataType::BOOL:
        return DataType::kBr_DataType_U1;
      case phi::DataType::INT64:
        return DataType::kBr_DataType_S64;
      case phi::DataType::INT32:
        return DataType::kBr_DataType_S32;
      // TODO... add more datatype mapping case.
      default: {
        std::stringstream sstream;
        sstream << dtype;
        PADDLE_THROW(::phi::errors::Unimplemented(
            "Unsupported phi::DataType: %s", sstream.str().c_str()));
        return DataType::kBr_DataType_Unknown;
      } break;
    }
  }

  Layout LayoutConvert(phi::DataLayout layout) {
    switch (layout) {
      case phi::DataLayout::NHWC:
        return Layout::kBr_Layout_NHWC;
      case phi::DataLayout::NCHW:
        return Layout::kBr_Layout_NCHW;
      // TODO... add more layout convert case.
      default:
        PADDLE_THROW(::phi::errors::Unimplemented(
            "Unsupported phi::DataLayout: %d", layout));
        return Layout::kBr_Layout_Unknown;
    }
  }

  shared_ptr<sucl::OpExecutor> op_executor_;
};

}  // namespace br_device

#endif  // BR_PADDLE_SUPA_H_
