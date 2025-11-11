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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename Context>
inline std::vector<size_t> GetLodFromRoisNum(const Context& dev_ctx,
                                             const phi::DenseTensor* rois_num) {
  std::vector<size_t> rois_lod;
  auto* rois_num_data = rois_num->data<int>();
  phi::DenseTensor cpu_tensor;
  phi::Copy<Context>(dev_ctx, *rois_num, phi::CPUPlace(), true, &cpu_tensor);
  rois_num_data = cpu_tensor.data<int>();
  rois_lod.push_back(static_cast<size_t>(0));
  for (int i = 0; i < rois_num->numel(); ++i) {
    rois_lod.push_back(rois_lod.back() + static_cast<size_t>(rois_num_data[i]));
  }
  return rois_lod;
}

template <typename T, typename Context>
void DistributeFpnProposalsKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& fpn_rois,
    const paddle::optional<phi::DenseTensor>& rois_num,
    int min_level,
    int max_level,
    int refer_level,
    int refer_scale,
    bool pixel_offset,
    std::vector<phi::DenseTensor*> multi_fpn_rois,
    std::vector<phi::DenseTensor*> multi_level_rois_num,
    phi::DenseTensor* restore_index) {
  VLOG(4) << "CALL SDAA DistributeFpnProposalsKernel";

  int num_level = max_level - min_level + 1;

  // check that the fpn_rois is not empty
  if (!rois_num.get_ptr()) {
    PADDLE_ENFORCE_EQ(
        fpn_rois.lod().size(),
        1UL,
        phi::errors::InvalidArgument("DistributeFpnProposalsOp needs LoD"
                                     "with one level"));
  }

  std::vector<size_t> fpn_rois_lod;
  if (rois_num.get_ptr()) {
    fpn_rois_lod =
        custom_kernel::GetLodFromRoisNum<Context>(dev_ctx, rois_num.get_ptr());
  } else {
    fpn_rois_lod = fpn_rois.lod().back();
  }
  int lod_size = fpn_rois_lod.size() - 1;
  int roi_num = fpn_rois_lod[lod_size];

  phi::DenseTensor rois_num_cpu, rois_num_sdaa;
  if (!rois_num.get_ptr()) {
    // get rois_num from fpn_rois_lod
    rois_num_cpu.Resize({lod_size});
    int* rois_num_cpu_data = dev_ctx.template HostAlloc<int>(&rois_num_cpu);

    for (int i = 0; i < lod_size; i++) {
      rois_num_cpu_data[i] = fpn_rois_lod[i + 1] - fpn_rois_lod[i];
    }
    // copy rois_num to sdaa
    rois_num_sdaa.Resize({lod_size});
    dev_ctx.template Alloc<int>(&rois_num_sdaa);
    phi::Copy(dev_ctx, rois_num_cpu, dev_ctx.GetPlace(), true, &rois_num_sdaa);

  } else {
    rois_num_sdaa = *(rois_num.get_ptr());
  }

  const int BoxDim = 4;
  std::vector<int> fpn_rois_dims = phi::vectorize<int>(fpn_rois.dims());
  std::vector<int> rois_num_dims = {lod_size};
  std::vector<void*> multi_fpn_rois_ptr, multi_level_rois_num_ptr;
  std::vector<tecodnnTensorDescriptor_t> multi_fpn_rois_Desc,
      multi_level_rois_num_Desc;

  std::vector<size_t> workSpaceSizeInBytesArray(num_level * lod_size);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t fpn_rois_Desc = sdaa_ops::GetTecodnnTensorDesc(
      fpn_rois_dims, fpn_rois.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t rois_num_Desc = sdaa_ops::GetTecodnnTensorDesc(
      rois_num_dims, DataType::INT32, TensorFormat::Undefined);

  // This function returns a 1-D array of size num_level * lod_size，where
  // workSpaceSizeInBytesArray[i] = M * 4 * sizeof(size_t).
  TECODNN_CHECK(tecodnnGetDistributeFPNProposalsWorkspaceSize(
      tecodnnHandle,
      min_level,
      max_level,
      refer_level,
      refer_scale,
      pixel_offset,
      fpn_rois_Desc,
      fpn_rois.data(),
      rois_num_Desc,
      rois_num_sdaa.data(),
      workSpaceSizeInBytesArray.data()));

  std::vector<std::vector<size_t>> lod_offset(
      num_level, std::vector<size_t>(lod_size + 1, 0));
  for (size_t i = 0; i < num_level; i++) {
    size_t step = 0;
    for (size_t j = 0; j < lod_size; j++) {
      lod_offset[i][j + 1] =
          lod_offset[i][j] +
          workSpaceSizeInBytesArray[i + step] / (4 * sizeof(size_t));
      step += num_level;
    }
  }

  // Alloc space
  for (size_t i = 0; i < num_level; i++) {
    size_t level_rois_num = lod_offset[i].back();

    std::vector<int> each_fpn_rois_dims = {static_cast<int>(level_rois_num),
                                           BoxDim};
    multi_fpn_rois[i]->Resize(phi::make_ddim(each_fpn_rois_dims));

    void* each_fpn_rois_ptr = dev_ctx.template Alloc<T>(multi_fpn_rois[i]);
    multi_fpn_rois_ptr.push_back(each_fpn_rois_ptr);

    tecodnnTensorDescriptor_t each_fpn_rois_Desc =
        sdaa_ops::GetTecodnnTensorDesc(each_fpn_rois_dims,
                                       multi_fpn_rois[i]->dtype(),
                                       TensorFormat::Undefined);
    multi_fpn_rois_Desc.push_back(each_fpn_rois_Desc);

    phi::LoD lod;
    lod.clear();
    lod.emplace_back(lod_offset[i]);
    phi::DenseTensorMeta lod_meta = {multi_fpn_rois[i]->dtype(),
                                     multi_fpn_rois[i]->dims(),
                                     multi_fpn_rois[i]->layout(),
                                     lod};
    multi_fpn_rois[i]->set_meta(lod_meta);
  }

  phi::DenseTensor multi_level_rois_num_ptr_tensor;
  void* multi_level_rois_num_ptr_tensor_data_ptr = nullptr;
  if (multi_level_rois_num.size() > 0) {
    std::vector<int> multi_level_rois_num_dims = {lod_size};
    for (int i = 0; i < num_level; i++) {
      multi_level_rois_num[i]->Resize(
          phi::make_ddim(multi_level_rois_num_dims));
      void* each_level_rois_num_ptr =
          dev_ctx.template Alloc<int>(multi_level_rois_num[i]);

      multi_level_rois_num_ptr.push_back(each_level_rois_num_ptr);

      tecodnnTensorDescriptor_t each_level_rois_num_Desc =
          sdaa_ops::GetTecodnnTensorDesc(multi_level_rois_num_dims,
                                         multi_level_rois_num[i]->dtype(),
                                         TensorFormat::Undefined);
      multi_level_rois_num_Desc.push_back(each_level_rois_num_Desc);
    }

    int multi_level_rois_num_ptr_size =
        multi_level_rois_num_ptr.size() * sizeof(void*);
    multi_level_rois_num_ptr_tensor.Resize(
        phi::make_ddim({multi_level_rois_num_ptr_size}));
    dev_ctx.template Alloc<int8_t>(&multi_level_rois_num_ptr_tensor);

    AsyncMemCpyH2D(
        nullptr,
        static_cast<C_Stream>(dev_ctx.stream()),
        reinterpret_cast<void*>(multi_level_rois_num_ptr_tensor.data()),
        reinterpret_cast<void*>(multi_level_rois_num_ptr.data()),
        multi_level_rois_num_ptr_size);
    multi_level_rois_num_ptr_tensor_data_ptr =
        multi_level_rois_num_ptr_tensor.data();
  }

  std::vector<int> restore_index_dims = {roi_num, 1};
  restore_index->Resize(phi::make_ddim(restore_index_dims));
  dev_ctx.template Alloc<int>(restore_index);

  tecodnnTensorDescriptor_t restore_index_Desc = sdaa_ops::GetTecodnnTensorDesc(
      restore_index_dims, restore_index->dtype(), TensorFormat::Undefined);

  int multi_fpn_rois_ptr_size = multi_fpn_rois_ptr.size() * sizeof(void*);
  phi::DenseTensor multi_fpn_rois_ptr_tensor;
  multi_fpn_rois_ptr_tensor.Resize(phi::make_ddim({multi_fpn_rois_ptr_size}));
  dev_ctx.template Alloc<int8_t>(&multi_fpn_rois_ptr_tensor);
  AsyncMemCpyH2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 reinterpret_cast<void*>(multi_fpn_rois_ptr_tensor.data()),
                 reinterpret_cast<void*>(multi_fpn_rois_ptr.data()),
                 multi_fpn_rois_ptr_size);

  TECODNN_CHECK(tecodnnDistributeFPNProposals(
      tecodnnHandle,
      min_level,
      max_level,
      refer_level,
      refer_scale,
      pixel_offset,
      fpn_rois_Desc,
      fpn_rois.data(),
      rois_num_Desc,
      rois_num_sdaa.data(),
      multi_fpn_rois_Desc.data(),
      reinterpret_cast<void**>(multi_fpn_rois_ptr_tensor.data()),
      multi_level_rois_num_Desc.data(),
      reinterpret_cast<void**>(multi_level_rois_num_ptr_tensor_data_ptr),
      restore_index_Desc,
      restore_index->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(fpn_rois_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(rois_num_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(restore_index_Desc));

  for (int i = 0; i < multi_fpn_rois_Desc.size(); i++) {
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(multi_fpn_rois_Desc[i]));
  }
  for (int i = 0; i < multi_level_rois_num_Desc.size(); i++) {
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(multi_level_rois_num_Desc[i]));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(distribute_fpn_proposals,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::DistributeFpnProposalsKernel,
                          float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
