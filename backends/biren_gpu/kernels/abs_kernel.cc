#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"

using namespace br_device;

namespace supa {

// Abs forward kernel
template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(1) << "Enter AbsKernel.";
  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(Abs)(), {&x}, {out});
  runner.Run();
  VLOG(1) << "Leave AbsKernel.";
}

// AbsGrad kernel
template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                   const phi::DenseTensor& dout, phi::DenseTensor* dx) {
  VLOG(1) << "Enter AbsGrad.";
  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(AbsGrad)(), {&x, &dout},
                                  {dx});
  runner.Run();
  VLOG(1) << "Leave AbsGrad.";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(abs, SUPA, ALL_LAYOUT, supa::AbsKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(abs_grad, SUPA, ALL_LAYOUT, supa::AbsGradKernel,
                          float) {}
