#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"
using namespace br_device;

namespace supa {

// Scale forward kernel
template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                 const phi::Scalar& scale, float bias, bool bias_after_scale,
                 phi::DenseTensor* out) {
  VLOG(1) << "Enter ScaleKernel.";
  OP_PARAMS(Scale) param;
  param.scale_ = scale.to<float>();
  param.bias_ = bias;
  param.bias_after_scale_ = bias_after_scale;
  SupaOpRunner<T, Context> runner(dev_ctx, param, {&x}, {out});
  runner.Run();
  VLOG(1) << "Leave ScaleKernel.";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(scale, SUPA, ALL_LAYOUT, supa::ScaleKernel, float,
                          int64_t) {}
