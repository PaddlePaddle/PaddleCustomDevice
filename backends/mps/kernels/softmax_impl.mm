#include "softmax_impl.h"
#include <Foundation/Foundation.h>
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

    int length = x_shape[0];
    NSArray<NSNumber *> *input_shape = @[ @(length) ];

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
  }
}

}  // namespace mps_kernel
