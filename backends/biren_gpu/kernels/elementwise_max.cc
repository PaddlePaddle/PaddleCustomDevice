#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"

using namespace br_device;

namespace supa {

template <typename T, typename Context>
void ElementwiseMaxKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                          const phi::DenseTensor& y, phi::DenseTensor* out) {
  VLOG(1) << "Enter ElementwiseMaxKernel.";

  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(ElementwiseMax)(),
                                  {&x, &y}, {out});
  runner.Run();

  VLOG(1) << "Leave ElementwiseMaxKernel.";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(maximum, SUPA, ALL_LAYOUT, supa::ElementwiseMaxKernel,
                          float) {}