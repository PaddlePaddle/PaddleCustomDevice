cmake_minimum_required(VERSION 3.10)

project(paddle-custom-intel-gpu CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_ONEDNN "use oneDNN with custom kernels" OFF)
option(WITH_KERNELS "compile with custom kernels" ON)
option(WITH_TESTING "compile with unit testing" ON)

if(NOT DEFINED ONEAPI_PATH)
  set(ONEAPI_PATH "$ENV{HOME}/intel/oneapi")
endif()

if(NOT EXISTS "${ONEAPI_PATH}")
  message(FATAL_ERROR "Can't find openapi directory ${ONEAPI_PATH}")
endif()

message(STATUS "OneAPI DIR : ${ONEAPI_PATH}")

if(NOT DEFINED ONEAPI_COMPILER_DIR)
  set(ONEAPI_COMPILER_DIR "compiler/latest")
endif()

if(NOT EXISTS "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}")
  message(
    FATAL_ERROR "Can't find directory ${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}")
endif()

find_package(PythonInterp 3.9 REQUIRED)
find_package(PythonLibs 3.9 REQUIRED)

if(NOT DEFINED ENV{DPL_ROOT})
  message(FATAL_ERROR "Please set the env var DPL_ROOT!" $ENV{DPL_ROOT})
endif()

set(DPCPP_COMPIER_PATH "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/bin/dpcpp")
set(ONEAPI_SYCL_INCLUDE
    "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/include/sycl/")
set(ONEAPI_SYCL_LIBDIR "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/lib/")
set(CMAKE_CXX_COMPILER "${DPCPP_COMPIER_PATH}")

# TODO(Zhiwei35): when using oneDNN, should add oneDNN include dir.
if(WITH_ONEDNN)
  include_directories(${ONEAPI_SYCL_INCLUDE} ${ONEDNN_INC})
else()
  include_directories(${ONEAPI_SYCL_INCLUDE} $ENV{DPL_ROOT}/include)
endif()

set(PLUGIN_NAME "paddle-custom-intel-gpu")
set(PLUGIN_VERSION "0.0.1")

include(paddle)
message(STATUS "PADDLE_BUILD_DIR DIR : ${PADDLE_BUILD_DIR}")
message(STATUS "ONEDNN_INC DIR : ${ONEDNN_INC}")

include_directories(${PADDLE_INC_DIR} ${CMAKE_SOURCE_DIR}
                    ${THIRD_PARTY_INC_DIR} ${CMAKE_SOURCE_DIR}/kernels)
link_directories(${PADDLE_LIB_DIR} ${ONEAPI_SYCL_LIBDIR})

# build onednn from source
if(WITH_ONEDNN)
  add_definitions(-DONEDNN)
endif()

add_definitions(-std=c++14)

# custom runtime
set(PLUGIN_SRCS runtime/runtime.cc)
add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)

# custom kernels
if(WITH_KERNELS)
  # TODO(Zhiwei35, to be more general when adding many kernels)
  list(APPEND PLUGIN_SRCS kernels/mean_kernel.cc)
  message(STATUS "CUSTOM_KERNEL_SRCS : ${CUSTOM_KERNEL_SRCS}")
  message(STATUS "PLUGIN_SRCS : ${PLUGIN_SRCS}")
endif()

# build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})

# link paddle shared library
include(third_party)
add_dependencies(${PLUGIN_NAME} third_party)
target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB})

if("${CMAKE_CXX_COMPILER}" STREQUAL "${DPCPP_COMPIER_PATH}")
  add_definitions(-DUSEDPCPP)
  set_target_properties(${PLUGIN_NAME} PROPERTIES CXX_STANDARD 17)
endif()

# testing
if(WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle")
  enable_testing()
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
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
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
  COMMENT "Creating plugin dirrectories------>>>")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${PLUGIN_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
