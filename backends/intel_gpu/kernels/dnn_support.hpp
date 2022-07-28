#pragma once
#include <CL/sycl.hpp>

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


}