# build graph executor
proto_library(framework_proto SRCS framework.proto)
# file(GLOB GRAPH_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
# file(GLOB_RECURSE GRAPH_ADAPTER_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} adapter/*.cc)
# add_library(graph_executor STATIC ${RELATIVE} ${CMAKE_CURRENT_BINARY_DIR}/framework.pb.cc ${GRAPH_SRCS} ${GRAPH_ADAPTER_SRCS})
# add_dependencies(graph_executor framework_proto extern_protobuf)
# target_link_libraries(graph_executor PRIVATE graph ge_runner protobuf)
# target_compile_options(graph_executor PRIVATE -fPIC)
# target_compile_definitions(graph_executor PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)

add_library(ge_c_api STATIC ge_c_api.cc)
target_link_libraries(ge_c_api PRIVATE graph ge_runner)
target_compile_options(ge_c_api PRIVATE -fPIC)
target_compile_definitions(ge_c_api PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
