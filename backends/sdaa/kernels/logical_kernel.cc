// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {
#define _dologicalop(type)                                                \
  if (x.dims().size() == 0 && y.dims().size() == 0) {                     \
    phi::DenseTensor x_temp(x), y_temp(y);                                \
    x_temp.Resize(phi::make_ddim({1}));                                   \
    y_temp.Resize(phi::make_ddim({1}));                                   \
    out->Resize(phi::make_ddim({1}));                                     \
    sdaa_ops::doLogicalOpTensor(                                          \
        dev_ctx, x_temp, y_temp, LogicalOpType::type, out);               \
  } else {                                                                \
    sdaa_ops::doLogicalOpTensor(dev_ctx, x, y, LogicalOpType::type, out); \
  }

template <typename T, typename Context>
void LogicalAndKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA LogicalAndKernel";
  dev_ctx.template Alloc<bool>(out);
  _dologicalop(And);
}

template <typename T, typename Context>
void LogicalOrKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA LogicalOrKernel";
  dev_ctx.template Alloc<bool>(out);
  _dologicalop(Or);
}

template <typename T, typename Context>
void LogicalXorKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA LogicalXorKernel";
  dev_ctx.template Alloc<bool>(out);
  _dologicalop(Xor);
}
#undef _dologicalop

template <typename T, typename Context>
void LogicalNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA LogicalNotKernel";
  dev_ctx.template Alloc<bool>(out);
  if (x.dims().size() == 0) {
    phi::DenseTensor x_temp(x);
    x_temp.Resize(phi::make_ddim({1}));
    out->Resize(phi::make_ddim({1}));
    sdaa_ops::doLogicalNotOpTensor(dev_ctx, x_temp, out);
  } else {
    sdaa_ops::doLogicalNotOpTensor(dev_ctx, x, out);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    logical_and, sdaa, ALL_LAYOUT, custom_kernel::LogicalAndKernel, int, bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_PLUGIN_KERNEL(
    logical_or, sdaa, ALL_LAYOUT, custom_kernel::LogicalOrKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_PLUGIN_KERNEL(
    logical_xor, sdaa, ALL_LAYOUT, custom_kernel::LogicalXorKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_PLUGIN_KERNEL(logical_not,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LogicalNotKernel,
                          int,
                          float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
