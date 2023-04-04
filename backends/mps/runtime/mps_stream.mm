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

#include "mps_stream.h"

namespace mps {

#define USE_COMMIT_AND_CONTINUE 1

MPSStream::MPSStream() {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  _serialQueue = dispatch_queue_create("metal gpu stream", nullptr);
  _executionDescriptor = [MPSGraphExecutionDescriptor new];
  _executionDescriptor.completionHandler =
      ^(NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *resultsDictionary,
        NSError *_Nullable error) {
      };
}

MPSStream::~MPSStream() {
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];

  assert(_commandBuffer == nil);
}

MPSCommandBuffer *MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }

  return _commandBuffer;
}

void MPSStream::synchronize(SyncType syncType) {
  if (!_commandBuffer) return;
  switch (syncType) {
    case SyncType::NONE:
      // typically in GPU to GPU copies we won't commit explicitly
      break;
    case SyncType::COMMIT:
      flush();
      break;
    case SyncType::COMMIT_AND_WAIT:
      commitAndWait();
      break;
    case SyncType::COMMIT_AND_CONTINUE:
      commitAndContinue();
      break;
  }
}

void MPSStream::commit(bool doFlush) {
#if USE_COMMIT_AND_CONTINUE
  [commandBuffer() commitAndContinue];
#else
  if (doFlush) {
    flush();
  }
#endif
}

void MPSStream::commitAndWait() {
  assert(_commandBuffer);
  [_commandBuffer commit];
  [_commandBuffer waitUntilCompleted];
  [_commandBuffer release];
  _commandBuffer = nil;
}

void MPSStream::commitAndContinue() {
  assert(_commandBuffer);
  [_commandBuffer commitAndContinue];
}

void MPSStream::flush() {
  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer release];
    _commandBuffer = nil;
  }
}

void MPSStream::_flush(bool commitAndWait) const {
  assert(_commandBuffer);
  [_commandBuffer commit];
  if (commitAndWait) {
    [_commandBuffer waitUntilCompleted];
  }
  [_commandBuffer release];
}

void MPSStream::executeMPSGraph(MPSGraph *mpsGraph,
                                NSDictionary *feeds,
                                NSDictionary *results,
                                SyncType syncType) {
  dispatch_sync(_serialQueue, ^() {
#if USE_COMMIT_AND_CONTINUE
    [mpsGraph encodeToCommandBuffer:commandBuffer()
                              feeds:feeds
                   targetOperations:nil
                  resultsDictionary:results
                executionDescriptor:_executionDescriptor];
    // mostly the syncType is NONE, but in some cases we may want to sync and
    // wait (e.g., gatherViewTensor)
    synchronize(syncType);
#else
    commit(true);
    [mpsGraph runAsyncWithMTLCommandQueue:_commandQueue
                                    feeds:feeds
                         targetOperations:nil
                        resultsDictionary:results
                      executionDescriptor:_executionDescriptor];
#endif
  });
}

class MPSStreamImpl {
 public:
  static MPSStream *getInstance();

 private:
  static MPSStream *_stream;
  MPSStreamImpl();
};

MPSStream *MPSStreamImpl::_stream = nullptr;

MPSStream *MPSStreamImpl::getInstance() {
  if (_stream == nullptr) {
    _stream = new MPSStream();
  }
  return _stream;
}

MPSStreamImpl::MPSStreamImpl() {}

MPSStream *getCurrentMPSStream() { return getDefaultMPSStream(); }

MPSStream *getDefaultMPSStream() { return MPSStreamImpl::getInstance(); }

}  // namespace mps
