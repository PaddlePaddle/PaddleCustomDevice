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

#include "activation_impl.h"
#include <Foundation/Foundation.h>
#include "glog/logging.h"
#include "mps_stream.h"
#include "op_utils.h"

namespace mps_kernel {

void Activation(const float *x, float *out, const std::vector<int64_t> &dims, ActivationOP op) {
  VLOG(5) << "mps_kernel::Activation start";
  mps::MPSStream *stream = mps::getCurrentMPSStream();

  @autoreleasepool {
    MPSGraph *mpsGraph = mps::make_mps_graph();

    MPSShape *x_shape = mps::vector_2_nsarray(dims);

    MPSGraphTensor *x_tensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, x_shape);

    MPSGraphTensor *output_tensor = nil;

    switch (op) {
      case ActivationOP::SIGMOID:
        output_tensor = [mpsGraph sigmoidWithTensor:x_tensor name:nil];
        break;
      case ActivationOP::EXP:
        output_tensor = [mpsGraph exponentWithTensor:x_tensor name:nil];
        break;
      case ActivationOP::SIN:
        output_tensor = [mpsGraph sinWithTensor:x_tensor name:nil];
        break;
      case ActivationOP::COS:
        output_tensor = [mpsGraph cosWithTensor:x_tensor name:nil];
        break;
      default:
        LOG(FATAL) << "Unsupported activation op type";
    }
    id<MTLBuffer> x_buffer = (id<MTLBuffer>)x;
    id<MTLBuffer> out_buffer = (id<MTLBuffer>)out;

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      x_tensor : [[[MPSGraphTensorData alloc] initWithMTLBuffer:x_buffer
                                                          shape:x_shape
                                                       dataType:MPSDataTypeFloat32] autorelease]
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      output_tensor :
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                   shape:x_shape
                                                dataType:MPSDataTypeFloat32] autorelease]
    };

    runMPSGraph(stream, mpsGraph, feeds, results);
  }
  VLOG(5) << "mps_kernel::Activation done";
}

void Pow(const float *x, float *out, const std::vector<int64_t> &dims, float factor) {
  VLOG(5) << "mps_kernel::POW start";
  mps::MPSStream *stream = mps::getCurrentMPSStream();

  @autoreleasepool {
    MPSGraph *mpsGraph = mps::make_mps_graph();

    MPSShape *x_shape = mps::vector_2_nsarray(dims);

    MPSGraphTensor *x_tensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, x_shape);
    MPSGraphTensor *factor_tensor = [mpsGraph constantWithScalar:(double)factor
                                                        dataType:MPSDataTypeFloat32];

    MPSGraphTensor *output_tensor = [mpsGraph powerWithPrimaryTensor:x_tensor
                                                     secondaryTensor:factor_tensor
                                                                name:nil];

    id<MTLBuffer> x_buffer = (id<MTLBuffer>)x;
    id<MTLBuffer> out_buffer = (id<MTLBuffer>)out;

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      x_tensor : [[[MPSGraphTensorData alloc] initWithMTLBuffer:x_buffer
                                                          shape:x_shape
                                                       dataType:MPSDataTypeFloat32] autorelease]
    };

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      output_tensor :
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                   shape:x_shape
                                                dataType:MPSDataTypeFloat32] autorelease]
    };

    runMPSGraph(stream, mpsGraph, feeds, results);
  }
  VLOG(5) << "mps_kernel::POW done";
}

}  // namespace mps_kernel
