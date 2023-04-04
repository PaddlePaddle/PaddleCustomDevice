#include "op_utils.h"

namespace mps {

MPSGraph *make_mps_graph() {
  MPSGraph *mpsGraph = [[MPSGraph new] autorelease];
  return mpsGraph;
}

MPSGraphTensor *mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph,
                                          MPSDataType dataType,
                                          MPSShape *mpsShape) {
  return [mpsGraph placeholderWithShape:mpsShape dataType:dataType name:nil];
}

void runMPSGraph(MPSStream *mpsStream,
                 MPSGraph *mpsGraph,
                 NSDictionary *feeds,
                 NSDictionary *results) {
  mpsStream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT_AND_WAIT);
}

}  // namespace mps
