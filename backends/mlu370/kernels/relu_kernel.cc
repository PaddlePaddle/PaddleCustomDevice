#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_RELU, 1.0);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::Active(dev_ctx,
                  act_desc.get(),
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(relu,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ReluKernel,
                          float,
                          phi::dtype::float16) {}
