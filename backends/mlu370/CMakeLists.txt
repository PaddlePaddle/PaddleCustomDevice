cmake_minimum_required(VERSION 3.10)

project(paddle-custom-mlu CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include(external/neuware)
include_directories(${CMAKE_SOURCE_DIR})

option(WITH_KERNELS    "compile with custom kernels"      ON)

set(CUSTOM_MLU_NAME        "paddle-custom-mlu")
set(CUSTOM_MLU_VERSION     "0.0.1")

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

# custom runtime
set(CUSTOM_MLU_SRCS runtime/runtime.cc)
add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
# TODO(qiil93): avoid compile error, to be removed
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)

# custom kernels
if (WITH_KERNELS)
  file(GLOB_RECURSE CUSTOM_KERNEL_SRCS RELATIVE ${CMAKE_SOURCE_DIR} kernels/*.cc)
  list(APPEND CUSTOM_MLU_SRCS ${CUSTOM_KERNEL_SRCS})
endif()

# C++ infer lib need to compatiable with CXX11
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

# build shared library
add_library(${CUSTOM_MLU_NAME} SHARED ${CUSTOM_MLU_SRCS})
target_link_libraries(${CUSTOM_MLU_NAME} PRIVATE ${NEUWARE_LIBS})

# link paddle shared library
include(third_party)
add_dependencies(${CUSTOM_MLU_NAME} third_party)
target_link_libraries(${CUSTOM_MLU_NAME} PRIVATE ${PADDLE_CORE_LIB})

find_package(PythonInterp ${PY_VERSION} REQUIRED)
find_package(PythonLibs ${PY_VERSION} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIR})
target_link_libraries(${CUSTOM_MLU_NAME} PRIVATE pybind)
