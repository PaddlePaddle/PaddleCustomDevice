// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "tcpx/tcpx.h"

namespace teco_paddle {
namespace tcpx {

namespace detail {
inline bool RangeEnabled() {
  static bool is_enabled = [] {
    const char* env = getenv("TECO_PADDLE_TCPX_RANGES");
    return env && atoi(env) != 0;
  }();
  return is_enabled;
}

inline bool RangeDetailEnabled() {
  static bool has_been_called = false;
  static bool is_enabled = [] {
    const char* env = getenv("TECO_PADDLE_TCPX_RANGES_DETAILED");
    return env && atoi(env) != 0;
  }();
  if (!has_been_called && is_enabled) {
    static bool dump_info_status = [] {
      const char* env = getenv("TECO_ENABLE_DUMP_INFO");
      return env && atoi(env) != 0;
    }();
    if (!dump_info_status) {
      LOG(WARNING) << "Please export TECO_ENABLE_DUMP_INFO=1 if you want to "
                      "get detailed ops information by TCPX!";
    }
  }
  has_been_called = true;
  return is_enabled;
}

std::string GetTblasRangeMessage(std::string op_name, std::string add_info);
std::string GetOpName(std::string sdaa_op);
}  // namespace detail

class TcpxDomain {
 public:
  explicit TcpxDomain(const char* name) : handle_(tcpxDomainCreate(name)) {}
  ~TcpxDomain() { tcpxDomainDestroy(handle_); }

  operator tcpxDomainHandle_t() const { return handle_; }

 private:
  tcpxDomainHandle_t handle_;
  TcpxDomain(const TcpxDomain&) = delete;
  TcpxDomain& operator=(const TcpxDomain&) = delete;
};

template <typename Subclass>
struct DomainBase {
  static tcpxDomainHandle_t GetTcpxDomain() {
    static TcpxDomain domain(Subclass::kName);
    return domain;
  }
};

struct NullDomain;

template <>
struct DomainBase<NullDomain> {
  static tcpxDomainHandle_t GetTcpxDomain() { return nullptr; }
};

struct NullDomain : public DomainBase<NullDomain> {};

struct CoreDomain : public DomainBase<CoreDomain> {
  static constexpr const char* kName = "teco-paddle-tcpx-core";
};

template <typename Domain = NullDomain>
class ScopedRangeIfEnabled {
 public:
  static tcpxDomainHandle_t domain() { return Domain::GetTcpxDomain(); }

  explicit ScopedRangeIfEnabled(std::function<std::string()> msg_fn) {
    if (!detail::RangeEnabled()) return;
    std::string msg = msg_fn();
    if (domain()) {
      tcpxDomainRangePush(domain(), msg.c_str());
    } else {
      tcpxRangePush(msg.c_str());
    }
  }

  void addExtraInfo(std::string info) {
    if (info != "{}") {
      extra_infos_ = info;
    }
  }

  ~ScopedRangeIfEnabled() {
    if (!detail::RangeEnabled()) return;

    if (detail::RangeDetailEnabled() && !extra_infos_.empty()) {
      std::string add_info = "__ADD_INFO::";
      add_info += extra_infos_;
      tcpxDomainMark(domain(), add_info.c_str());
    }

    if (domain()) {
      tcpxDomainRangePop(domain());
    } else {
      tcpxRangePop();
    }
  }
  std::string extra_infos_{};
};

}  // namespace tcpx
}  // namespace teco_paddle
