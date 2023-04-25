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

#include "elementwise_impl.h"
#include <Foundation/Foundation.h>
#include "glog/logging.h"
#include "mps_stream.h"
#include "op_utils.h"

namespace mps_kernel {

void Elementwise(const float* x,
                 const float* y,
                 float* out,
                 const std::vector<int64_t>& dims,
                 MPSElementwiseOP op) {
  VLOG(5) << "mps_kernel::Elementwise start";
  mps::MPSStream* stream = mps::getCurrentMPSStream();

  @autoreleasepool {
    MPSGraph* mpsGraph = mps::make_mps_graph();

    MPSShape* x_shape = mps::vector_2_nsarray(dims);
    MPSShape* y_shape = mps::vector_2_nsarray(dims);

    MPSGraphTensor* x_tensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, x_shape);
    MPSGraphTensor* y_tensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, y_shape);

    MPSGraphTensor* output_tensor = nil;

    switch (op) {
      case MPSElementwiseOP::ADD:
        output_tensor = [mpsGraph additionWithPrimaryTensor:x_tensor
                                            secondaryTensor:y_tensor
                                                       name:nil];
        break;
      case MPSElementwiseOP::SUB:
        output_tensor = [mpsGraph subtractionWithPrimaryTensor:x_tensor
                                               secondaryTensor:y_tensor
                                                          name:nil];
        break;
      case MPSElementwiseOP::MUL:
        output_tensor = [mpsGraph multiplicationWithPrimaryTensor:x_tensor
                                                  secondaryTensor:y_tensor
                                                             name:nil];
        break;
      case MPSElementwiseOP::DIV:
        output_tensor = [mpsGraph divisionWithPrimaryTensor:x_tensor
                                            secondaryTensor:y_tensor
                                                       name:nil];
        break;
      default:
        LOG(FATAL) << "Unsupported elementwise op";
    }

    id<MTLBuffer> x_buffer = (id<MTLBuffer>)x;
    id<MTLBuffer> y_buffer = (id<MTLBuffer>)y;
    id<MTLBuffer> out_buffer = (id<MTLBuffer>)out;

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      x_tensor : [[[MPSGraphTensorData alloc] initWithMTLBuffer:x_buffer
                                                          shape:x_shape
                                                       dataType:MPSDataTypeFloat32] autorelease],
      y_tensor : [[[MPSGraphTensorData alloc] initWithMTLBuffer:y_buffer
                                                          shape:y_shape
                                                       dataType:MPSDataTypeFloat32] autorelease]
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      output_tensor :
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                   shape:x_shape
                                                dataType:MPSDataTypeFloat32] autorelease]
    };

    mps::runMPSGraph(stream, mpsGraph, feeds, results);
  }
  VLOG(5) << "mps_kernel::Elementwise end";
}

}  // namespace mps_kernel
