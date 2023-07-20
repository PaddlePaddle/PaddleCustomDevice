#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"

using namespace br_device;

namespace supa {

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(1) << "Enter ExpKernel.";
  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(Exp)(), {&x}, {out});
  runner.Run();
  VLOG(1) << "Leave ExpKernel.";
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx, const phi::DenseTensor& out,
                   const phi::DenseTensor& dout, phi::DenseTensor* dx) {
  VLOG(1) << "Enter ExpGradKernel.";
  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(ExpGrad)(), {&out, &dout},
                                  {dx});
  VLOG(1) << "Leave ExpGradKernel.";
  runner.Run();
}
}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(exp, SUPA, ALL_LAYOUT, supa::ExpKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(exp_grad, SUPA, ALL_LAYOUT, supa::ExpGradKernel,
                          float) {}