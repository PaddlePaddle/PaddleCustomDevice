set(PROTO_FRAMEWORK_FILE
    "${PADDLE_SOURCE_DIR}/paddle/phi/core/framework/framework.proto")
get_filename_component(PROTO_FRAMEWORK_WE "${PROTO_FRAMEWORK_FILE}" NAME_WE)

set(GENERATED_FRAMEWORK_SRC
    "${CMAKE_CURRENT_BINARY_DIR}/paddle/phi/core/framework/${PROTO_FRAMEWORK_WE}.pb.cc"
)
set(GENERATED_FRAMEWORK_HDR
    "${CMAKE_CURRENT_BINARY_DIR}/paddle/phi/core/framework/${PROTO_FRAMEWORK_WE}.pb.h"
)

message(
  STATUS "FRAMEWORK_WE - CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(
  OUTPUT "${GENERATED_FRAMEWORK_SRC}" "${GENERATED_FRAMEWORK_HDR}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${CMAKE_CURRENT_BINARY_DIR}/paddle/phi/core/"
  COMMAND
    ${PROTOBUF_PROTOC_EXECUTABLE} -I${PADDLE_SOURCE_DIR}/paddle/phi/core/
    --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/paddle/phi/core
    ${PROTO_FRAMEWORK_FILE}
  DEPENDS "${PROTO_FRAMEWORK_FILE}"
  COMMENT "Generating C++ protocol buffer for ${PROTO_FRAMEWORK_FILE}"
  VERBATIM)

add_custom_command(
  OUTPUT
    "${PADDLE_SOURCE_DIR}/paddle/phi/core/${PROTO_FRAMEWORK_WE}.pb.cc"
    "${PADDLE_SOURCE_DIR}/paddle/phi/core/framework/${PROTO_FRAMEWORK_WE}.pb.h"
  COMMAND ${CMAKE_COMMAND} -E copy ${GENERATED_FRAMEWORK_SRC}
          ${PADDLE_SOURCE_DIR}/paddle/phi/core/${PROTO_FRAMEWORK_WE}.pb.cc
  COMMAND
    ${CMAKE_COMMAND} -E copy ${GENERATED_FRAMEWORK_HDR}
    ${PADDLE_SOURCE_DIR}/paddle/phi/core/framework/${PROTO_FRAMEWORK_WE}.pb.h
  DEPENDS ${GENERATED_FRAMEWORK_SRC} ${GENERATED_FRAMEWORK_HDR}
  COMMENT "Copy framework.pb.xxx")

add_library(framework_proto STATIC
            "${PADDLE_SOURCE_DIR}/paddle/phi/core/${PROTO_FRAMEWORK_WE}.pb.cc")
target_include_directories(framework_proto PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(framework_proto PUBLIC protobuf)
set_target_properties(framework_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
