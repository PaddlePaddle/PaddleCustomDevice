#include "glog/logging.h"
#include "kernels/funcs/br_paddle_supa.h"
#include "paddle/phi/extension.h"

using namespace br_device;

namespace supa {

template <typename T, typename Context>
void ElementwiseMinKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                          const phi::DenseTensor& y, phi::DenseTensor* out) {
  VLOG(1) << "Enter ElementwiseMinKernel.";

  SupaOpRunner<T, Context> runner(dev_ctx, OP_PARAMS(ElementwiseMin)(),
                                  {&x, &y}, {out});
  runner.Run();

  VLOG(1) << "Leave ElementwiseMinKernel.";
}

}  // namespace supa

PD_REGISTER_PLUGIN_KERNEL(minimum, SUPA, ALL_LAYOUT, supa::ElementwiseMinKernel,
                          float) {}