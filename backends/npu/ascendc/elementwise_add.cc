// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ascendc/elementwise_add.h"

#include "kernel_operator.h"  // NOLINT

namespace custom_device::ascendc {

constexpr uint64_t MaxOprandBytes = 65536;
constexpr uint64_t MinOprandBytes = 32;

// implementation of kernel function
template <typename T, uint64_t BUFFER_NUM = 1>
class KernelAdd {
 public:
  __aicore__ inline KernelAdd() {}
  __aicore__ inline void Init(__gm__ uint8_t* x,
                              __gm__ uint8_t* y,
                              __gm__ uint8_t* z,
                              uint64_t length) {
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    tileLength_ = MaxOprandBytes / sizeof(T);
    totalLength_ = __align_up(length, BUFFER_NUM * tileLength_);

    // get start index for current core, core parallel
    xGm.SetGlobalBuffer((__gm__ T*)x, totalLength_);
    yGm.SetGlobalBuffer((__gm__ T*)y, totalLength_);
    zGm.SetGlobalBuffer((__gm__ T*)z, totalLength_);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueX, BUFFER_NUM, MaxOprandBytes);
    pipe.InitBuffer(inQueueY, BUFFER_NUM, MaxOprandBytes);
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, MaxOprandBytes);
  }

  __aicore__ inline void Process() {
    for (uint64_t i = AscendC::GetBlockIdx();
         i * BUFFER_NUM * tileLength_ < totalLength_;
         i += AscendC::GetBlockNum()) {
#pragma unroll
      for (uint64_t j = 0; j < BUFFER_NUM; ++j) {
        auto progress = i * BUFFER_NUM + j;
        this->CopyIn(progress);
        this->Compute(progress);
        this->CopyOut(progress);
      }
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress) {
    // alloc tensor from queue memory
    AscendC::LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inQueueY.template AllocTensor<T>();
    // copy progress_th tile from global tensor to local tensor
    AscendC::DataCopy(xLocal, xGm[progress * tileLength_], tileLength_);
    AscendC::DataCopy(yLocal, yGm[progress * tileLength_], tileLength_);
    // enque input tensors to VECIN queue
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }

  __aicore__ inline void Compute(int32_t progress) {
    // deque input tensors from VECIN queue
    AscendC::LocalTensor<T> xLocal = inQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inQueueY.template DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outQueueZ.template AllocTensor<T>();
    // call Add instr for computation
    AscendC::Add(zLocal, xLocal, yLocal, tileLength_);
    // enque the output tensor to VECOUT queue
    outQueueZ.template EnQue<T>(zLocal);
    // free input tensors for reuse
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress) {
    // deque output tensor from VECOUT queue
    AscendC::LocalTensor<T> zLocal = outQueueZ.template DeQue<T>();
    // copy progress_th tile from local tensor to global tensor
    AscendC::DataCopy(zGm[progress * tileLength_], zLocal, tileLength_);
    // free output tensor for reuse
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<T> xGm, yGm, zGm;

  uint64_t tileLength_;
  uint64_t totalLength_;
};

// NOTE(wangran16): global functions cannot be defined as template function
__global__ __aicore__ void __elementwise_add_f32(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t length, uint64_t buffer_num) {
  if (buffer_num == 1) {
    KernelAdd<float, 1> op;
    op.Init(x, y, z, length);
    op.Process();
  } else {
    KernelAdd<float, 2> op;
    op.Init(x, y, z, length);
    op.Process();
  }
}

__global__ __aicore__ void __elementwise_add_f16(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t length, uint64_t buffer_num) {
  if (buffer_num == 1) {
    KernelAdd<half, 1> op;
    op.Init(x, y, z, length);
    op.Process();
  } else {
    KernelAdd<half, 2> op;
    op.Init(x, y, z, length);
    op.Process();
  }
}

__global__ __aicore__ void __elementwise_add_s32(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t length, uint64_t buffer_num) {
  if (buffer_num == 1) {
    KernelAdd<int32_t, 1> op;
    op.Init(x, y, z, length);
    op.Process();
  } else {
    KernelAdd<int32_t, 2> op;
    op.Init(x, y, z, length);
    op.Process();
  }
}

__global__ __aicore__ void __elementwise_add_s16(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t length, uint64_t buffer_num) {
  if (buffer_num == 1) {
    KernelAdd<int16_t, 1> op;
    op.Init(x, y, z, length);
    op.Process();
  } else {
    KernelAdd<int16_t, 2> op;
    op.Init(x, y, z, length);
    op.Process();
  }
}

}  // namespace custom_device::ascendc

extern "C" void elementwise_add_f32(uint32_t blockDim,
                                    void* l2ctrl,
                                    void* stream,
                                    const void* x,
                                    const void* y,
                                    void* z,
                                    uint64_t length,
                                    uint64_t buffer_num) {
  custom_device::ascendc::__elementwise_add_f32<<<blockDim, l2ctrl, stream>>>(
      reinterpret_cast<uint8_t*>(const_cast<void*>(x)),
      reinterpret_cast<uint8_t*>(const_cast<void*>(y)),
      reinterpret_cast<uint8_t*>(z),
      length,
      buffer_num);
}

extern "C" void elementwise_add_f16(uint32_t blockDim,
                                    void* l2ctrl,
                                    void* stream,
                                    const void* x,
                                    const void* y,
                                    void* z,
                                    uint64_t length,
                                    uint64_t buffer_num) {
  custom_device::ascendc::__elementwise_add_f16<<<blockDim, l2ctrl, stream>>>(
      reinterpret_cast<uint8_t*>(const_cast<void*>(x)),
      reinterpret_cast<uint8_t*>(const_cast<void*>(y)),
      reinterpret_cast<uint8_t*>(z),
      length,
      buffer_num);
}

extern "C" void elementwise_add_s32(uint32_t blockDim,
                                    void* l2ctrl,
                                    void* stream,
                                    const void* x,
                                    const void* y,
                                    void* z,
                                    uint64_t length,
                                    uint64_t buffer_num) {
  custom_device::ascendc::__elementwise_add_s32<<<blockDim, l2ctrl, stream>>>(
      reinterpret_cast<uint8_t*>(const_cast<void*>(x)),
      reinterpret_cast<uint8_t*>(const_cast<void*>(y)),
      reinterpret_cast<uint8_t*>(z),
      length,
      buffer_num);
}

extern "C" void elementwise_add_s16(uint32_t blockDim,
                                    void* l2ctrl,
                                    void* stream,
                                    const void* x,
                                    const void* y,
                                    void* z,
                                    uint64_t length,
                                    uint64_t buffer_num) {
  custom_device::ascendc::__elementwise_add_s16<<<blockDim, l2ctrl, stream>>>(
      reinterpret_cast<uint8_t*>(const_cast<void*>(x)),
      reinterpret_cast<uint8_t*>(const_cast<void*>(y)),
      reinterpret_cast<uint8_t*>(z),
      length,
      buffer_num);
}
