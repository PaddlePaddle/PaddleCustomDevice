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

void Softmax(const float *in, float *out, const std::vector<int64_t> &dims, int axis) {
  VLOG(5) << "mps_kernel::Softmax start";
  mps::MPSStream *stream = mps::getCurrentMPSStream();
  @autoreleasepool {
    MPSGraph *mpsGraph = mps::make_mps_graph();

    NSArray *shape = mps::vector_2_nsarray(dims);

    MPSGraphTensor *inputTensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, shape);
    MPSGraphTensor *outputTensor = [mpsGraph softMaxWithTensor:inputTensor axis:axis name:nil];

    id<MTLBuffer> in_buffer = (id<MTLBuffer>)in;
    id<MTLBuffer> out_buffer = (id<MTLBuffer>)out;

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      inputTensor : [[[MPSGraphTensorData alloc] initWithMTLBuffer:in_buffer
                                                             shape:shape
                                                          dataType:MPSDataTypeFloat32] autorelease]
    };
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      outputTensor : [[[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                              shape:shape
                                                           dataType:MPSDataTypeFloat32] autorelease]
    };

    runMPSGraph(stream, mpsGraph, feeds, results);
  }
  VLOG(5) << "mps_kernel::Softmax done";
}

void SoftmaxGrad(const float *out,
                 const float *out_grad,
                 const std::vector<int64_t> &dims,
                 int axis,
                 float *in_grad) {
  mps::MPSStream *stream = mps::getCurrentMPSStream();
  @autoreleasepool {
    MPSGraph *mpsGraph = mps::make_mps_graph();

    MPSShape *shape = mps::vector_2_nsarray(dims);

    MPSGraphTensor *softmaxTensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, shape);
    MPSGraphTensor *outputGradTensor =
        mps::mpsGraphRankedPlaceHolder(mpsGraph, MPSDataTypeFloat32, shape);

    MPSGraphTensor *mulTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                          secondaryTensor:outputGradTensor
                                                                     name:nil];
    MPSGraphTensor *mulSumTensor = [mpsGraph reductionSumWithTensor:mulTensor
                                                               axis:(NSInteger)axis
                                                               name:nil];
    MPSGraphTensor *gradSubTensor = [mpsGraph subtractionWithPrimaryTensor:outputGradTensor
                                                           secondaryTensor:mulSumTensor
                                                                      name:nil];

    MPSGraphTensor *inputGradTensor = [mpsGraph multiplicationWithPrimaryTensor:softmaxTensor
                                                                secondaryTensor:gradSubTensor
                                                                           name:nil];
    id<MTLBuffer> out_buffer = (id<MTLBuffer>)out;
    id<MTLBuffer> out_grad_buffer = (id<MTLBuffer>)out_grad;
    id<MTLBuffer> in_grad_buffer = (id<MTLBuffer>)in_grad;

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
      softmaxTensor :
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                   shape:shape
                                                dataType:MPSDataTypeFloat32] autorelease],
      outputGradTensor :
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:out_grad_buffer
                                                   shape:shape
                                                dataType:MPSDataTypeFloat32] autorelease]
    };
    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
      inputGradTensor :
          [[[MPSGraphTensorData alloc] initWithMTLBuffer:in_grad_buffer
                                                   shape:shape
                                                dataType:MPSDataTypeFloat32] autorelease]
    };
    runMPSGraph(stream, mpsGraph, feeds, results);
    VLOG(5) << "SoftmaxGrad done";
  }
}

}  // namespace mps_kernel
