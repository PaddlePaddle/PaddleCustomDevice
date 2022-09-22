macro(UNSET_VAR VAR_NAME)
  unset(${VAR_NAME} CACHE)
  unset(${VAR_NAME})
endmacro()

# graph executor dependencies
include(external/zlib)
include(external/xxhash)
set(TEMP ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0)
include(external/protobuf)
set(CMAKE_CXX_FLAGS ${TEMP})

# build graph executor
proto_library(framework_proto SRCS framework.proto)
file(GLOB_RECURSE GRAPH_SRCS RELATIVE ${CMAKE_SOURCE_DIR} graph/*.cc)
file(GLOB_RECURSE GRAPH_ADAPTER_SRCS RELATIVE ${CMAKE_SOURCE_DIR} graph/adapter/*.cc)
add_library(graph_executor STATIC ${RELATIVE} ${CMAKE_CURRENT_BINARY_DIR}/framework.pb.cc ${GRAPH_SRCS} ${GRAPH_ADAPTER_SRCS})
add_dependencies(graph_executor framework_proto extern_protobuf)
target_link_libraries(graph_executor PRIVATE graph ge_runner protobuf)
target_compile_options(graph_executor PRIVATE -fPIC)
target_compile_definitions(graph_executor PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
