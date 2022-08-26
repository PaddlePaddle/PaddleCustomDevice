#pragma once
#include <CL/sycl.hpp>
#include <thread>
#include <vector>

#include "oneapi/dnnl/dnnl_sycl.hpp"
#define show_msg(title, x)                                                    \
  std::cout << "[" << title << "][" << std::hex << std::this_thread::get_id() \
            << std::dec << "]["<< __FILE__<< ":"<< __LINE__ <<"]: " << x << std::endl;

// #define show(x) show_msg("SHOW", x)
#define show(x)
#define show_kernel(x) show_msg("KERNEL", x)

namespace dnn_support {

template <class T>
struct toDnnType {};

template <>
struct toDnnType<int> {
  const static dnnl::memory::data_type type = dnnl::memory::data_type::s32;
};

template <>
struct toDnnType<float> {
  const static dnnl::memory::data_type type = dnnl::memory::data_type::f32;
};

template <>
struct toDnnType<char> {
  const static dnnl::memory::data_type type = dnnl::memory::data_type::bf16;
};

/*
template <>
struct toDnnType<double> {
  const static dnnl::memory::data_type type = dnnl::memory::data_type::f64;
};

*/

}

