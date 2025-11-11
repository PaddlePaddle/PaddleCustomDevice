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

#pragma once

#include <Foundation/Foundation.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "runtime/mps_device.h"

typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLCommandBuffer> MTLCommandBuffer_t;
typedef id<MTLDevice> MTLDevice_t;

namespace mps {

enum class SyncType {
  NONE,
  COMMIT,
  COMMIT_AND_WAIT,
  COMMIT_AND_CONTINUE,
};

class MPSStream {
 public:
  MPSStream(const MPSStream &other) = delete;
  void operator=(const MPSStream &) = delete;

  static MPSStream *getInstance();

  ~MPSStream();
  MTLCommandQueue_t commandQueue() const { return _commandQueue; }
  dispatch_queue_t queue() const { return _serialQueue; }

  MPSCommandBuffer *commandBuffer();
  void commit(bool flush);
  void commitAndWait();
  void commitAndContinue();
  void synchronize(SyncType syncType);
  void flush();
  void executeMPSGraph(MPSGraph *mpsGraph,
                       NSDictionary *feeds,
                       NSDictionary *results,
                       SyncType syncType = SyncType::NONE);

  MTLCommandQueue_t stream() const { return _commandQueue; }

 private:
  MPSStream();
  MTLCommandQueue_t _commandQueue = nil;
  MPSCommandBuffer *_commandBuffer = nil;
  MPSGraphExecutionDescriptor *_executionDescriptor = nil;
  void _flush(bool commitAndWait) const;
  dispatch_queue_t _serialQueue = nullptr;
};

MPSStream *getCurrentMPSStream();

MPSStream *getDefaultMPSStream();

}  // namespace mps
