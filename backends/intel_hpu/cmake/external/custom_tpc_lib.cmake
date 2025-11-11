set(DOWNLOAD_URL "https://paddle-ci.cdn.bcebos.com/libcustom_tpc_perf_lib.so")
set(TARGET_DIR "${CMAKE_BINARY_DIR}/python/paddle_custom_device/intel_hpu")
set(TARGET_PATH "${TARGET_DIR}/libcustom_tpc_perf_lib.so")

file(MAKE_DIRECTORY ${TARGET_DIR})
file(DOWNLOAD ${DOWNLOAD_URL} ${TARGET_PATH} STATUS download_status)

list(GET download_status 0 download_success)
if(NOT (download_success EQUAL 0))
  message(FATAL_ERROR "Failed to download ${DOWNLOAD_URL} to ${TARGET_PATH}")
endif()

message(STATUS "Downloaded ${DOWNLOAD_URL} to ${TARGET_PATH}")
