#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"

using namespace br_device;

namespace supa {
template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                       float slope, float offset, phi::DenseTensor* out) {
  OP_PARAMS(HardSigmoid) param;
  param.slope_ = slope;
  param.offset_ = offset;
  SupaOpRunner<T, Context> runner(dev_ctx, param, {&x}, {out});
  runner.Run();
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx, const phi::DenseTensor& out,
                           const phi::DenseTensor& dout, float slope,
                           float offset, phi::DenseTensor* dx) {
  OP_PARAMS(HardSigmoidGrad) param;
  param.slope_ = slope;
  param.offset_ = offset;
  SupaOpRunner<T, Context> runner(dev_ctx, param, {&out, &dout}, {dx});
  runner.Run();
}
}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(hard_sigmoid, SUPA, ALL_LAYOUT,
                          supa::HardSigmoidKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(hard_sigmoid_grad, SUPA, ALL_LAYOUT,
                          supa::HardSigmoidGradKernel, float) {}
