cmake_minimum_required(VERSION 3.10)

project(paddle-ascend CXX C)

option(WITH_TESTING    "compile plugin with unit testing"        OFF)
option(WITH_KERNELS    "build custom kernels"                    OFF)

set(PLUGIN_NAME        "paddle_ascend")
set(PLUGIN_VERSION      "0.0.1")

set(PADDLE_INC_DIR     "${PYTHON_SITE_PACKGES}/paddle/include/")
set(PADDLE_LIB_DIR     "${PYTHON_SITE_PACKGES}/paddle/fluid/")

set(NPU_INC_DIR        "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/include/")
set(NPU_LIB_DIR        "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/")

set(PLUGIN_SRCS runtime/runtime.cc)
set(INCLUDE_DIR ${PADDLE_INC_DIR} ${NPU_INC_DIR})

if (WITH_KERNELS)
  set(BOOST_INC_DIR      "${PADDLE_ROOT}/build/third_party/boost/src/extern_boost")
  set(EIGEN_INC_DIR      "${PADDLE_ROOT}/build/third_party/eigen3/src/extern_eigen3/")
  set(GFLAGS_INC_DIR     "${PADDLE_ROOT}/build/third_party/install/gflags/include")
  set(GLOG_INC_DIR       "${PADDLE_ROOT}/build/third_party/install/glog/include")
  set(THREAD_INC_DIR     "${PADDLE_ROOT}/build/third_party/threadpool/src/extern_threadpool")

  set(THIRD_PARTY_INC_DIR ${BOOST_INC_DIR} ${EIGEN_INC_DIR} ${GFLAGS_INC_DIR} ${GLOG_INC_DIR} ${THREAD_INC_DIR})
  set(RUNTIME_INC_DIR ${CMAKE_SOURCE_DIR}/runtime)

  file(GLOB PLUGIN_KERNEL_FILES RELATIVE ${CMAKE_SOURCE_DIR} kernels/*.cc)
  list(APPEND PLUGIN_SRCS ${PLUGIN_KERNEL_FILES})

  list(APPEND INCLUDE_DIR ${THIRD_PARTY_INC_DIR} ${RUNTIME_INC_DIR})

  find_file(CORE_AVX_FOUND core_avx.so ${PADDLE_LIB_DIR})
  if (CORE_AVX_FOUND)
    set(CORE_LIB ":core_avx.so")
  else()
    set(CORE_LIB ":core_noavx.so")
  endif()

  add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)  # for out CustomContext
  add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)  # for out fluid seperate
endif()

include_directories(${INCLUDE_DIR})
link_directories(${PADDLE_LIB_DIR} ${NPU_LIB_DIR})

####### build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
target_link_libraries(${PLUGIN_NAME} PRIVATE ascendcl)
target_link_libraries(${PLUGIN_NAME} PRIVATE acl_op_compiler)
if (WITH_KERNELS)
 target_link_libraries(${PLUGIN_NAME} PRIVATE ${CORE_LIB})  # special name
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

####### packing wheel package
add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMENT "Creating plugin dirrectories------>>>"
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>"
)

add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
