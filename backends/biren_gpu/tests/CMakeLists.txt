function(py_test_modules TARGET_NAME)
  set(options SERIAL)
  set(oneValueArgs "")
  set(multiValueArgs MODULES DEPS ENVS)
  cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  add_test(
    NAME ${TARGET_NAME}
    COMMAND
      ${CMAKE_COMMAND} -E env
      CUSTOM_DEVICE_ROOT=${CMAKE_BINARY_DIR}/python/paddle_custom_device/
      PYTHONPATH=${PYTHON_SOURCE_DIR}:${PYTHON_SOURCE_DIR}/tests:$ENV{PYTHONPATH}
      ${py_test_modules_ENVS} python ${PYTHON_SOURCE_DIR}/tools/test_runner.py
      ${py_test_modules_MODULES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if(py_test_modules_SERIAL)
    set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
  endif()
endfunction()

add_subdirectory(unittests)
