add_library(ge_c_api STATIC ge_c_api.cc)
target_link_libraries(ge_c_api PRIVATE graph ge_runner)
target_compile_options(ge_c_api PRIVATE -fPIC)
target_compile_definitions(ge_c_api PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
