if("${TEST_CUSTOM_DEVICE_ROOT}" STREQUAL "")
  return()
endif()

add_executable(test_runtime test_runtime.cc)
add_dependencies(test_runtime third_party)
target_link_libraries(test_runtime gtest gtest_main ${SDAA_LIB} ${TECODNN_LIB})
add_test(test_runtime test_runtime)

py_test_modules(test_profiler MODULES test_profiler ENVS ENABLE_SDPTI=0)
py_test_modules(
  test_profiler_with_kernel MODULES test_profiler_with_kernel ENVS
  FLAGS_allocator_strategy=naive_best_fit FLAGS_init_allocated_mem=True)
py_test_modules(
  test_profiler_with_kernel_with_env
  MODULES
  test_profiler_with_kernel
  ENVS
  FLAGS_allocator_strategy=naive_best_fit
  FLAGS_init_allocated_mem=True
  FLAGS_sdaa_error_check=True
  ENABLE_SDPTI=1)
