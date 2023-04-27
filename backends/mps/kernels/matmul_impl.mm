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

#include "matmul_impl.h"
#include <Foundation/Foundation.h>
#include "glog/logging.h"
#include "mps_stream.h"
#include "op_utils.h"

namespace mps_kernel {

void Matmul(const float* x,
            const float* y,
            float* out,
            const std::vector<int64_t>& x_dims,
            const std::vector<int64_t>& y_dims,
            bool transpose_x,
            bool transpose_y) {
  VLOG(5) << "mps_kernel::Matmul start";
  mps::MPSStream* stream = mps::getCurrentMPSStream();
  @autoreleasepool {
    MPSGraph* mpsGraph = mps::make_mps_graph();

    MPSShape* x_shape = mps::vector_2_nsarray(x_dims);
    MPSShape* y_shape = mps::vector_2_nsarray(y_dims);

    MPSGraphTensor* x_tensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, x_shape);
    MPSGraphTensor* y_tensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, y_shape);

    MPSGraphTensor* t1 = nil;
    MPSGraphTensor* t2 = nil;

    if (transpose_x) {
      t1 = [mpsGraph transposeTensor:x_tensor dimension:-1 withDimension:-2 name:nil];
    } else {
      t1 = x_tensor;
    }

    if (transpose_y) {
      t2 = [mpsGraph transposeTensor:y_tensor dimension:-1 withDimension:-2 name:nil];
    } else {
      t2 = y_tensor;
    }

    MPSGraphTensor* output_tensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:t1
                                                                    secondaryTensor:t2
                                                                               name:nil];

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

    runMPSGraph(stream, mpsGraph, feeds, results);
  }
  VLOG(5) << "mps_kernel::Matmul done";
}

}  // namespace mps_kernel
