# Given a cu_file (e.g. foo/bar.cu) relative to CMAKE_CURRENT_SOURCE_DIR
# and a eigen_target, create a cpp file that includes the .cu file, and set
# ${cpp_file_var} in the parent scope to the full path of the new file. The new
# file will be generated in:
# ${CMAKE_CURRENT_BINARY_DIR}/<eigen_target_prefix>/${cu_file}.cpp
function(eigen_wrap_cu_in_cpp cpp_file_var cu_file eigen_target)
  # eigen_get_target_property(prefix ${eigen_target} PREFIX)
  set(wrapped_source_file "${CMAKE_CURRENT_SOURCE_DIR}/${cu_file}")
  set(cpp_file "${CMAKE_CURRENT_BINARY_DIR}/${eigen_target}_cppwrap/${cu_file}.cpp")
  configure_file("${Eigen_SOURCE_DIR}/cmake/wrap_source_file.cpp.in" "${cpp_file}")
  set(${cpp_file_var} "${cpp_file}" PARENT_SCOPE)
endfunction()


