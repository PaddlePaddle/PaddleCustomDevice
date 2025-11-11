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

#include <atomic>
#include <cmath>
#include <ctime>
#include <thread>

#include "gtest/gtest.h"
#include "sdaa_runtime.h"  // NOLINT
#include "tecodnn.h"       // NOLINT

unsigned int seed = 123;
const double eps = 1e-6;

float rand_in_range(float min, float max) {
  if (fabs(max - min) < eps) return min;
  return ((max - min) * (rand_r(&seed) / static_cast<float>(RAND_MAX)) + min);
}

// copy to host and check the result
template <typename T>
bool check_result(void *d_x, T *h, const int num) {
  T *h_x = static_cast<T *>(malloc(num * sizeof(T)));
  // make sure memcpy get the right result when stream is not blocking stream
  sdaaDeviceSynchronize();
  sdaaMemcpy(h_x, d_x, num * sizeof(T), sdaaMemcpyDeviceToHost);
  double max_diff = 0;
  for (int i = 0; i < num; i++) {
    max_diff = std::max(max_diff, fabs(h[i] - h_x[i]));
  }
  free(h_x);
  return max_diff < eps;
}

bool doScale(tecodnnHandle_t &tecodnnHandle,  // NOLINT
             void *x,
             float alpha,
             std::vector<int> shape) {
  if (shape.size() != 4) {
    std::cout << "The shape must be 4D!" << std::endl;
    return false;
  }
  tecodnnDataType_t dataType = TECODNN_DATA_FLOAT;
  const int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
  tecodnnTensorDescriptor_t xDesc;
  tecodnnCreateTensorDescriptor(&xDesc);
  tecodnnSetTensor4dDescriptorEx(xDesc, dataType, N, C, H, W, 0, 0, 0, 0);
  tecodnnScaleTensor(tecodnnHandle, xDesc, x, &alpha);
  tecodnnDestroyTensorDescriptor(xDesc);
  return true;
}

bool doArange(tecodnnHandle_t &tecodnnHandle,  // NOLINT
              void *out,
              int start,
              int end,
              int step,
              std::vector<int> shape) {
  if (shape.size() != 1) {
    std::cout << "The shape must be 1D!" << std::endl;
    return false;
  }
  tecodnnDataType_t dataType = TECODNN_DATA_INT32;
  const int N = shape.size();
  int dims_arr[N];
  for (int i = 0; i < N; ++i) dims_arr[i] = shape[i];
  tecodnnTensorDescriptor_t Desc;
  tecodnnCreateTensorDescriptor(&Desc);
  tecodnnSetTensorNdDescriptor(Desc, dataType, shape.size(), dims_arr, NULL);
  tecodnnArange(tecodnnHandle, &start, &end, &step, Desc, out);
  tecodnnDestroyTensorDescriptor(Desc);
  return true;
}

// do memoryh2d for multi-thread startup
void doMemoryH2D_thread(void *d_x, void *x, const int num, int device) {
  sdaaSetDevice(device);
  sdaaMemcpy(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice);
}

// do scale for multi-thread startup
// note: handle must be created in the thread, otherwise the calculation result
// will be inaccurate.
bool doScale_thread(void *x, float alpha, std::vector<int> shape, int device) {
  if (shape.size() != 4) {
    std::cout << "The shape must be 4D!" << std::endl;
    return false;
  }
  sdaaSetDevice(device);
  tecodnnHandle_t handle;
  tecodnnCreate(&handle);
  sdaaStream_t stream;
  sdaaStreamCreate(&stream);
  tecodnnSetStream(handle, stream);
  tecodnnDataType_t dataType = TECODNN_DATA_FLOAT;
  const int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
  tecodnnTensorDescriptor_t xDesc;
  tecodnnCreateTensorDescriptor(&xDesc);
  tecodnnSetTensor4dDescriptorEx(xDesc, dataType, N, C, H, W, 0, 0, 0, 0);
  tecodnnScaleTensor(handle, xDesc, x, &alpha);
  tecodnnDestroyTensorDescriptor(xDesc);
  sdaaStreamDestroy(stream);
  tecodnnDestroy(handle);
  return true;
}

// case 1.compute + memcpyD2H
TEST(test_memcpy, memcpy1) {
  const int num = 200;
  float *x = static_cast<float *>(malloc(num * sizeof(float)));
  for (int i = 0; i < num; ++i) x[i] = rand_in_range(0, 1);
  float alpha = 0.5;
  void *d_x;
  sdaaMalloc(&d_x, num * sizeof(float));
  sdaaMemcpy(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice);
  for (int i = 0; i < num; ++i) x[i] *= alpha;
  tecodnnHandle_t handle;
  tecodnnCreate(&handle);
  sdaaStream_t stream;
  sdaaStreamCreate(&stream);
  tecodnnSetStream(handle, stream);
  bool ret = doScale(handle, d_x, alpha, {5, 10, 2, 2});
  EXPECT_EQ(ret, true);
  ret = check_result(d_x, x, num);
  EXPECT_EQ(ret, true);
  sdaaFree(d_x);
  sdaaStreamDestroy(stream);
  tecodnnDestroy(handle);
  free(x);
}

// case 2.memcpyasync(D2H, D2D H2D)
TEST(test_memcpy, memcpyasync1) {
  const int num = 200;
  float *x = static_cast<float *>(malloc(num * sizeof(float)));
  float *y = static_cast<float *>(malloc(num * sizeof(float)));
  for (int i = 0; i < num; i++) x[i] = rand_in_range(0, 1);
  void *d_x;
  void *d_y;
  sdaaMalloc(&d_x, num * sizeof(float));
  sdaaMalloc(&d_y, num * sizeof(float));
  sdaaStream_t stream;
  sdaaStreamCreate(&stream);
  // test H2D Async
  sdaaMemcpyAsync(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice, stream);
  bool ret = check_result(d_x, x, num);
  EXPECT_EQ(ret, true);

  // test D2D Async
  sdaaMemcpyAsync(
      d_y, d_x, num * sizeof(float), sdaaMemcpyDeviceToDevice, stream);
  ret = check_result(d_y, x, num);
  EXPECT_EQ(ret, true);

  // test D2H Async
  // sdaamemcpyAsync d2h is synchronous when host memory is not pin_memory.
  sdaaMemcpyAsync(y, d_x, num * sizeof(float), sdaaMemcpyDeviceToHost, stream);
  double max_diff = 0;
  for (int i = 0; i < num; i++) {
    max_diff = std::max(max_diff, fabs(y[i] - x[i]));
  }
  EXPECT_LE(max_diff, eps);

  sdaaFree(d_x);
  sdaaFree(d_y);
  sdaaStreamDestroy(stream);
  free(x);
  free(y);
}

// case 3.whether memcpyasync block host
TEST(test_memcpy, memcpyasync2) {
  // element number is large enough to copy slowly.
  const int num = 1024 * 1024;
  float *x;
  // malloc pin_memory so that memcpyh2d should be asynchronous.
  sdaaMallocHost(reinterpret_cast<void **>(&x), num * sizeof(float));
  for (int i = 0; i < num; i++) x[i] = rand_in_range(0, 1);
  void *d_x;
  sdaaMalloc(&d_x, num * sizeof(float));
  sdaaStream_t stream;
  sdaaStreamCreate(&stream);
  // make sure all the jobs in stream is finished
  sdaaStreamSynchronize(stream);
  sdaaMemcpyAsync(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice, stream);
  auto ret = sdaaStreamQuery(stream);

  EXPECT_EQ(ret, sdaaErrorNotReady);

  sdaaFree(d_x);
  sdaaStreamDestroy(stream);
  sdaaFreeHost(static_cast<void *>(x));
}

// case 4. Event test
TEST(test_event, eventrecord1) {
  const int num = 10240;
  float *x = static_cast<float *>(malloc(num * sizeof(float)));
  for (int i = 0; i < num; ++i) x[i] = rand_in_range(0, 1);
  float alpha = 0.5;
  void *d_x;
  sdaaMalloc(&d_x, num * sizeof(float));
  sdaaMemcpy(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice);
  for (int i = 0; i < num; ++i) x[i] *= alpha;
  tecodnnHandle_t handle;
  tecodnnCreate(&handle);
  sdaaStream_t stream;
  sdaaStreamCreate(&stream);
  tecodnnSetStream(handle, stream);
  sdaaEvent_t start, stop;
  sdaaEventCreate(&start);
  sdaaEventCreate(&stop);
  sdaaEventRecord(start, stream);
  doScale(handle, d_x, alpha, {16, 10, 16, 4});
  sdaaEventRecord(stop, stream);

  sdaaEventSynchronize(stop);
  auto status = sdaaEventQuery(stop);
  EXPECT_EQ(status, sdaaSuccess);

  float elapsedTime;
  sdaaEventElapsedTime(&elapsedTime, start, stop);
  EXPECT_GE(elapsedTime, eps);

  sdaaEventDestroy(start);
  sdaaEventDestroy(stop);
  status = sdaaEventQuery(stop);
  EXPECT_EQ(status, sdaaErrorContextIsDestroyed);

  sdaaFree(d_x);
  sdaaStreamDestroy(stream);
  tecodnnDestroy(handle);
  free(x);
}

// case 5. Using events to synchronize streams
TEST(test_event, eventrecord2) {
  const int num = 10240;
  float *x = static_cast<float *>(malloc(num * sizeof(float)));
  for (int i = 0; i < num; ++i) x[i] = rand_in_range(0, 1);
  float alpha = 0.5;
  void *d_x;
  sdaaMalloc(&d_x, num * sizeof(float));
  sdaaMemcpy(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice);

  tecodnnHandle_t handle1;
  tecodnnCreate(&handle1);
  sdaaStream_t stream1;
  sdaaStreamCreate(&stream1);
  tecodnnSetStream(handle1, stream1);

  tecodnnHandle_t handle2;
  tecodnnCreate(&handle2);
  sdaaStream_t stream2;
  sdaaStreamCreate(&stream2);
  tecodnnSetStream(handle2, stream2);

  sdaaEvent_t event;
  sdaaEventCreate(&event);
  sdaaEventRecord(event, stream1);
  doScale(handle1, d_x, alpha, {16, 10, 16, 4});

  sdaaStreamWaitEvent(stream2, event, 0);
  doScale(handle2, d_x, alpha, {16, 10, 16, 4});

  for (int i = 0; i < num; ++i) x[i] *= alpha * alpha;
  bool ret = check_result(d_x, x, num);
  EXPECT_EQ(ret, true);

  sdaaEventDestroy(event);
  sdaaStreamDestroy(stream1);
  sdaaStreamDestroy(stream2);
  tecodnnDestroy(handle1);
  tecodnnDestroy(handle2);
  sdaaFree(d_x);
  free(x);
}

// case 6. test DeviceSynchronize
TEST(test_device, devicesync1) {
  const int num = 10240;
  float *x = static_cast<float *>(malloc(num * sizeof(float)));
  float *y = static_cast<float *>(malloc(num * sizeof(float)));
  for (int i = 0; i < num; ++i) x[i] = rand_in_range(0, 1);
  for (int i = 0; i < num; ++i) y[i] = rand_in_range(0, 1);
  float alpha = 0.5;
  int slave_num = 0;
  auto status = sdaaGetDeviceCount(&slave_num);
  EXPECT_EQ(status, sdaaSuccess);

  int original_num = 0;
  sdaaGetDevice(&original_num);

  int actual_num = slave_num - 1;
  status = sdaaSetDevice(actual_num);
  EXPECT_EQ(status, sdaaSuccess);

  int cur_num = 0;
  status = sdaaGetDevice(&cur_num);
  EXPECT_EQ(status, sdaaSuccess);
  EXPECT_EQ(cur_num, actual_num);

  sdaaSetDevice(original_num);

  void *d_x;
  sdaaMalloc(&d_x, num * sizeof(float));
  sdaaMemcpy(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice);
  void *d_y;
  sdaaMalloc(&d_y, num * sizeof(float));
  sdaaMemcpy(d_y, y, num * sizeof(float), sdaaMemcpyHostToDevice);

  tecodnnHandle_t handle1;
  tecodnnCreate(&handle1);
  sdaaStream_t stream1;
  sdaaStreamCreate(&stream1);
  tecodnnSetStream(handle1, stream1);

  tecodnnHandle_t handle2;
  tecodnnCreate(&handle2);
  sdaaStream_t stream2;
  sdaaStreamCreate(&stream2);
  tecodnnSetStream(handle2, stream2);

  doScale(handle1, d_x, alpha, {16, 10, 16, 4});
  doScale(handle2, d_y, alpha, {16, 10, 16, 4});

  for (int i = 0; i < num; ++i) {
    x[i] *= alpha;
    y[i] *= alpha;
  }
  sdaaDeviceSynchronize();
  bool ret = check_result(d_x, x, num);
  EXPECT_EQ(ret, true);
  ret = check_result(d_y, y, num);
  EXPECT_EQ(ret, true);

  sdaaStreamDestroy(stream1);
  sdaaStreamDestroy(stream2);
  tecodnnDestroy(handle1);
  tecodnnDestroy(handle2);
  sdaaFree(d_x);
  sdaaFree(d_y);
  free(x);
  free(y);
}

// case 7. test sdaaMalloc allign
TEST(test_memory, sdaamalloc1) {
  void *d_x;
  sdaaMalloc(&d_x, sizeof(float));
  uint64_t x = (uint64_t)d_x;
  uint64_t y = 0x2000;
  uint64_t ret = x % y;
  EXPECT_EQ(ret, 0);
  sdaaFree(d_x);

  sdaaMalloc(&d_x, 1024 * sizeof(float));
  x = (uint64_t)d_x;
  ret = x % y;
  EXPECT_EQ(ret, 0);
  sdaaFree(d_x);
}

// case 8. test memset block
TEST(test_memory, sdaamemset1) {
  const int num = 1024;
  int *x = static_cast<int *>(malloc(num * sizeof(int)));
  for (int i = 0; i < num; ++i) x[i] = i + 1;
  void *out;
  sdaaMalloc(&out, num * sizeof(int));
  tecodnnHandle_t handle;
  tecodnnCreate(&handle);
  sdaaStream_t stream;
  sdaaStreamCreate(&stream);
  tecodnnSetStream(handle, stream);
  sdaaMemset(out, 0, num * sizeof(int));
  bool ret = doArange(handle, out, 1, 1025, 1, {1024});
  EXPECT_EQ(ret, true);
  ret = check_result(out, x, num);
  EXPECT_EQ(ret, true);
  sdaaFree(out);
  sdaaStreamDestroy(stream);
  tecodnnDestroy(handle);
  free(x);
}

// case 9. test stream
TEST(test_stream, stream1) {
  const int num = 10240;
  float *x = static_cast<float *>(malloc(num * sizeof(float)));
  for (int i = 0; i < num; ++i) x[i] = rand_in_range(0, 1);
  float alpha = 0.5;
  void *d_x;
  sdaaMalloc(&d_x, num * sizeof(float));
  sdaaMemcpy(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice);

  tecodnnHandle_t handle1;
  tecodnnCreate(&handle1);
  sdaaStream_t stream1;
  sdaaStreamCreate(&stream1);
  tecodnnSetStream(handle1, stream1);

  tecodnnHandle_t handle2;
  tecodnnCreate(&handle2);
  sdaaStream_t stream2;
  sdaaStreamCreate(&stream2);
  tecodnnSetStream(handle2, stream2);

  doScale(handle1, d_x, alpha, {16, 10, 16, 4});

  while (sdaaStreamQuery(stream1) != sdaaSuccess) {
  }
  doScale(handle2, d_x, alpha, {16, 10, 16, 4});

  for (int i = 0; i < num; ++i) x[i] *= alpha * alpha;
  bool ret = check_result(d_x, x, num);
  EXPECT_EQ(ret, true);

  sdaaStreamDestroy(stream1);
  sdaaStreamDestroy(stream2);
  tecodnnDestroy(handle1);
  tecodnnDestroy(handle2);
  sdaaFree(d_x);
  free(x);
}

// case 10. dataloader memcpyh2d + compute
TEST(test_dataloader, dataloader1) {
  const int num = 10240;
  float *x = static_cast<float *>(malloc(num * sizeof(float)));
  for (int i = 0; i < num; ++i) x[i] = rand_in_range(0, 1);
  float alpha = 0.5;
  void *d_x;
  void *d_y;
  sdaaMalloc(&d_x, num * sizeof(float));
  sdaaMalloc(&d_y, num * sizeof(float));
  sdaaMemcpy(d_x, x, num * sizeof(float), sdaaMemcpyHostToDevice);

  int device = 0;
  sdaaGetDevice(&device);

  // test that copying to d_y and computing in d_x under multithreading
  std::thread t1(doMemoryH2D_thread, d_y, x, num, device);
  std::vector<int> shape = {16, 10, 16, 4};
  std::thread t2(doScale_thread, d_x, alpha, shape, device);
  t1.join();
  t2.join();
  for (int i = 0; i < num; ++i) x[i] *= alpha;
  bool ret = check_result(d_x, x, num);
  EXPECT_EQ(ret, true);

  sdaaFree(d_x);
  sdaaFree(d_y);
  free(x);
}
