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

NSArray<NSNumber *> *vector_2_nsarray(const std::vector<int64_t> &vec) {
  NSMutableArray *nsArray = [NSMutableArray array];
  for (int i = 0; i < vec.size(); i++) {
    NSNumber *number = [NSNumber numberWithLongLong:vec[i]];
    [nsArray addObject:number];
  }
  NSArray *array = [NSArray arrayWithArray:nsArray];
  return array;
}

}  // namespace mps
