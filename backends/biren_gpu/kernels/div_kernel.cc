#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"

using namespace br_device;

namespace supa {

// Div forward kernel
template <typename T, typename Context>
void DivKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               const phi::DenseTensor& y, phi::DenseTensor* out) {
  VLOG(4) << "Enter DivKernel";
  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(Div)(), {&x, &y}, {out});
  runner.Run();
  VLOG(4) << "Leave DivKernel";
}

template <typename T, typename Context>
void DivGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                   const phi::DenseTensor& y, const phi::DenseTensor& out,
                   const phi::DenseTensor& dout, int axis, phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  VLOG(4) << "Enter DivGradKernel";
  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(DivGrad)(),
                                  {&x, &y, &dout}, {dx, dy});
  runner.Run();
  VLOG(4) << "Leave DivGradKernel";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(divide, SUPA, ALL_LAYOUT, supa::DivKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(divide_grad, SUPA, ALL_LAYOUT, supa::DivGradKernel,
                          float) {}
