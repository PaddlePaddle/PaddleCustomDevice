cmake_minimum_required(VERSION 3.10)

project(paddle-custom-sdaa CXX C)
add_definitions(-std=c++14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_TESTING "compile with unit testing" ON)
option(NATIVE_SDAA "use native sdaa lib" ON)
option(WITH_MKLDNN "compile with MKLDNN support" ON)
option(WITH_SW "compile with sw arch support" OFF)
option(WITH_GIT_COMMIT "compile with git commit print" ON)
option(WITH_PADDLE_INFO "compile with paddle commit/version print" ON)

set(PLUGIN_NAME "paddle-custom-sdaa")
if(DEFINED ENV{PLUGIN_VERSION})
  set(PLUGIN_VERSION $ENV{PLUGIN_VERSION})
else()
  set(PLUGIN_VERSION "0.0.1")
endif()

include(paddle)
include(generic)
include(teco)
include(third_party)

include_directories(
  ${PADDLE_INC_DIR}
  ${TECODNN_INC}
  ${CMAKE_SOURCE_DIR}
  ${EXTEND_OP_INC}
  ${SDPTI_INC}
  ${TBLAS_INC}
  ${TECODNN_CUSTOM_INC}
  ${TCCL_INC}
  ${CLANG_INC})

if(NATIVE_SDAA)
  add_definitions(-DNATIVE_SDAA)
  include_directories(${SDAA_INC})
endif()

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
# TODO(Aganlengzi): avoid compile error, to be removed
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)
if(WITH_SW)
  add_definitions(-DPADDLE_WITH_SW)
endif()

if(WITH_GIT_COMMIT)
  execute_process(
    COMMAND git rev-parse --short HEAD
    OUTPUT_VARIABLE GIT_COMMIT_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  add_definitions(-DGIT_COMMIT_ID=\"${GIT_COMMIT_ID}\")
endif()

if(WITH_PADDLE_INFO)
  get_paddle_info(PADDLE_COMMIT_ID PADDLE_FULL_VERSION PADDLE_BUILD_ENV_PATH)
  add_definitions(-DPADDLE_COMMIT_ID=\"${PADDLE_COMMIT_ID}\")
  add_definitions(-DPADDLE_FULL_VERSION=\"${PADDLE_FULL_VERSION}\")
endif()

file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  kernels/*.cc)
set(DYNLOAD_SRCS dynload/sdpti.cc dynload/dynamic_loader.cc)
list(APPEND PLUGIN_SRCS ${DYNLOAD_SRCS})
file(
  GLOB_RECURSE PLUGIN_RUNTIME_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  runtime/*.cc)
list(APPEND PLUGIN_SRCS ${PLUGIN_RUNTIME_SRCS})

# version dump
set(VERSION_DUMP_TARGET version_dump)
set(PADDLE_SDAA_TOOLS_DIR ${CMAKE_SOURCE_DIR}/tools)
set(VERSION_QUERY_DIR ${PADDLE_SDAA_TOOLS_DIR}/version)
list(APPEND PLUGIN_SRCS ${VERSION_QUERY_DIR}/query.cc)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/version/minimum_supported_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/tools/version/minimum_supported_version.h @ONLY)
add_executable(${VERSION_DUMP_TARGET} ${VERSION_QUERY_DIR}/query.cc
                                      ${VERSION_QUERY_DIR}/dump.cc)
target_include_directories(${VERSION_DUMP_TARGET}
                           PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${VERSION_DUMP_TARGET} PRIVATE ${TCCL_LIB})
target_link_libraries(${VERSION_DUMP_TARGET} PRIVATE ${SDAA_LIB})
target_link_libraries(${VERSION_DUMP_TARGET} PRIVATE ${TECODNN_LIB})
target_link_libraries(${VERSION_DUMP_TARGET} PRIVATE ${TBLAS_LIB})
target_link_libraries(${VERSION_DUMP_TARGET} PRIVATE ${TECODNN_CUSTOM_LIB})

add_custom_command(
  TARGET ${VERSION_DUMP_TARGET}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Run version_dump to generate version.py"
  COMMAND ./${VERSION_DUMP_TARGET})

# build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
set_target_properties(
  ${PLUGIN_NAME} PROPERTIES LINK_FLAGS
                            "-Wl,-rpath,$ORIGIN/lib/,--enable-new-dtags")
target_include_directories(${PLUGIN_NAME} PRIVATE ${PADDLE_SDAA_TOOLS_DIR})

# link third_party
add_dependencies(${PLUGIN_NAME} third_party)
target_link_libraries(${PLUGIN_NAME} PRIVATE gflags glog)

# sdaa
if(NATIVE_SDAA)
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${SDAA_LIB})
endif()
target_link_libraries(${PLUGIN_NAME} PRIVATE ${EXTEND_OP_LIB})
target_link_libraries(${PLUGIN_NAME} PRIVATE ${TECODNN_LIB})
target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB})
target_link_libraries(${PLUGIN_NAME} PRIVATE ${TBLAS_LIB})
target_link_libraries(${PLUGIN_NAME} PRIVATE stdc++fs)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${TCCL_LIB})
target_link_libraries(${PLUGIN_NAME} PRIVATE ${TECODNN_CUSTOM_LIB})

# testing
if(WITH_TESTING)
  set(TEST_CUSTOM_DEVICE_ROOT
      CUSTOM_DEVICE_ROOT=${CMAKE_BINARY_DIR}/python/paddle_custom_device)
  add_subdirectory(tests)
endif()

# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(
  TARGET ${PLUGIN_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/sdaa-ext
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/lib/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating plugin dirrectories------>>>")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  # COMMAND ${CMAKE_COMMAND} -E make_directory
  # ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/sdaa COMMAND
  # ${CMAKE_COMMAND} -E make_directory
  # ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/sdaa/sdaa_ext
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/sdaa_ext/python/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${PLUGIN_NAME}
  COMMENT "Packing paddle-custom-sdaa whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
