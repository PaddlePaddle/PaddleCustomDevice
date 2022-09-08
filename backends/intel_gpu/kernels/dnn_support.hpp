#pragma once
#include <CL/sycl.hpp>
#include <thread>
#include <vector>

#include "oneapi/dnnl/dnnl_sycl.hpp"

namespace config {
enum SUBSYS_LEVELS { vError = 16 };
template<int v>
struct DeviceConfig {
  size_t chunk_size;
  size_t plugin_verbose;

  template <class T>
  T getEnvValue(const char* name, T defaultValue) {
    T ret = defaultValue;

    auto p = std::getenv(name);

    if (p) {
      std::stringstream ss;
      ss << p;
      ss >> ret;
      if (ss.fail()) {
        throw std::runtime_error("getEnvValue(): Can't convert type");
      }
    }

    return ret;
  }

  DeviceConfig() : chunk_size{4}, plugin_verbose{config::vError} {
    chunk_size = getEnvValue("PLUGIN_CHUNK_SIZE", chunk_size);
    plugin_verbose = getEnvValue("PLUGIN_VERBOSE", plugin_verbose);
    if(plugin_verbose)
    {
      plugin_verbose |= config::vError;
    }
  }
};

}


using DeviceConfig = config::DeviceConfig<0>;
using DeviceConfigPtr = std::unique_ptr<DeviceConfig>;
extern DeviceConfigPtr devconf;
extern std::mutex mx;

inline void InitializeDevConf() {
     if(!devconf)
     {
        std::lock_guard<decltype(mx)> l(mx);
        if(!devconf)
        {
          devconf = std::make_unique<DeviceConfig>();
        }
     }
}

template<class T>
const T* shortPath(const T* p) {

  const char* r = p;
  while(*p) {
    if(*p == '/') r=p;
    ++p;
  }
  return r;

}

#define show_msg(title, vbit, x) \
  if(devconf && devconf->plugin_verbose & vbit) {  \
  std::cout << "[" << title << "][" << std::hex << std::this_thread::get_id() \
            << std::dec << "]["<< shortPath(__FILE__)<< ":"<< __LINE__ <<"]: "<< x << std::endl; }

#define show_debug(x) show_msg("debug", 1, x)
#define show_memory(x) show_msg("mem", 2, x)
#define show_kernel(x) show_msg("kernel", 4, x)
#define show_error(x) show_msg("error", config::vError, x)

template <class T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
  o << "{ ";
  for(auto item : v)
  {
    o << item << ",";
  }
  o << " }";
  return o;
}

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
