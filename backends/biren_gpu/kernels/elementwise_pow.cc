#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"

using namespace br_device;

namespace supa {

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                          const phi::DenseTensor& y, phi::DenseTensor* out) {
  VLOG(1) << "Enter ElementwisePowKernel.";

  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(ElementwisePow)(),
                                  {&x, &y}, {out});
  runner.Run();

  VLOG(1) << "Leave ElementwisePowKernel.";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(elementwise_pow, SUPA, ALL_LAYOUT,
                          supa::ElementwisePowKernel, float) {}