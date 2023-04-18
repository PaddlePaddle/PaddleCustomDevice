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

#include "softmax_impl.h"
#include <Foundation/Foundation.h>
#include "glog/logging.h"
#include "mps_stream.h"
#include "op_utils.h"

namespace mps_kernel {

void Softmax(const float *in,
             float *out,
             std::vector<int64_t> x_shape,
             std::vector<int64_t> out_shape,
             int axis) {
  mps::MPSStream *stream = mps::getCurrentMPSStream();
  @autoreleasepool {
    MPSGraph *mpsGraph = mps::make_mps_graph();

    NSArray *input_shape = mps::shape2Array(x_shape);

    MPSGraphTensor *inputTensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, input_shape);
    MPSGraphTensor *outputTensor = [mpsGraph softMaxWithTensor:inputTensor axis:axis name:nil];

    id<MTLBuffer> in_buffer = (id<MTLBuffer>)in;
    id<MTLBuffer> out_buffer = (id<MTLBuffer>)out;

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      inputTensor : [[[MPSGraphTensorData alloc] initWithMTLBuffer:in_buffer
                                                             shape:input_shape
                                                          dataType:MPSDataTypeFloat32] autorelease]
    };
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      outputTensor : [[[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                              shape:input_shape
                                                           dataType:MPSDataTypeFloat32] autorelease]
    };

    runMPSGraph(stream, mpsGraph, feeds, results);
    VLOG(5) << "Softmax done";
  }
}

}  // namespace mps_kernel
