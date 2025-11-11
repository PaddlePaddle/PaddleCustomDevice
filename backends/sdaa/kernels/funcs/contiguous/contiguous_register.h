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

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

namespace sdaa_copy {

class ContiguousOpt {
 public:
  ContiguousOpt() {}
  virtual ~ContiguousOpt() = default;
  virtual bool Optimize(const Context& dev_ctx,
                        const phi::DenseTensor& src,
                        phi::DenseTensor* dst) = 0;
  virtual bool CanOptimize(const Context& dev_ctx,
                           const phi::DenseTensor& src,
                           phi::DenseTensor* dst) {
    return false;
  }
};

namespace register_opt {

enum ProfLevel {
  PROF_HIGH = 0,
  PROF_LOW = 1,
  PROF_MAX_CNT = 2,
};

class CopyOptRegister {
  using OptMap = std::map<std::string, std::unique_ptr<ContiguousOpt>>;

 public:
  ~CopyOptRegister() = default;
  static CopyOptRegister* GetInstance() {
    static CopyOptRegister instance;
    return &instance;
  }

  template <typename T>
  void Register(std::string& name,        // NOLINT
                std::unique_ptr<T>& ptr,  // NOLINT
                const ProfLevel level = ProfLevel::PROF_HIGH) {
    std::lock_guard<std::mutex> lock(mu_);
    registry[level].emplace(name, std::move(ptr));
  }

  bool CanOptimize(std::string& name,  // NOLINT
                   const Context& dev_ctx,
                   const phi::DenseTensor& src,
                   phi::DenseTensor* dst) {
    for (int8_t level = ProfLevel::PROF_HIGH; level < ProfLevel::PROF_MAX_CNT;
         level++) {
      if (FindOptimize(registry[level], name, dev_ctx, src, dst)) {
        return true;
      }
    }
    return false;
  }

  bool Run(const std::string& name,
           const Context& dev_ctx,
           const phi::DenseTensor& src,
           phi::DenseTensor* dst) {
    for (int8_t level = ProfLevel::PROF_HIGH; level < ProfLevel::PROF_MAX_CNT;
         level++) {
      auto itr = registry[level].find(name);
      if (itr != registry[level].end()) {
        return itr->second->Optimize(dev_ctx, src, dst);
      }
    }
    return false;
  }

 private:
  CopyOptRegister() {}
  mutable std::mutex mu_;
  mutable std::array<OptMap, ProfLevel::PROF_MAX_CNT> registry;

 private:
  bool FindOptimize(OptMap& opt_map,    // NOLINT
                    std::string& name,  // NOLINT
                    const Context& dev_ctx,
                    const phi::DenseTensor& src,
                    phi::DenseTensor* dst) {
    for (auto& opt : opt_map) {
      if (opt.second->CanOptimize(dev_ctx, src, dst)) {
        name = opt.first;
        return true;
      }
    }
    return false;
  }
};

class CopyOptBuilder {
 public:
  template <typename T>
  CopyOptBuilder(std::string name,
                 std::unique_ptr<T>& ptr,  // NOLINT
                 const ProfLevel level = ProfLevel::PROF_HIGH) {
    CopyOptRegister::GetInstance()->Register(name, ptr, level);
  }
  ~CopyOptBuilder() = default;
};
}  // namespace register_opt

#define REGISTER_COPY_OPT(name, Optimization) \
  REGISTER_COPY_OPT_INSTANCE(name, name, Optimization)
#define REGISTER_COPY_OPT_LOW(name, Optimization)                \
  auto contig_opt_##name = std::make_unique<Optimization>();     \
  static register_opt::CopyOptBuilder register_contig_opt##name( \
      #name, contig_opt_##name, register_opt::ProfLevel::PROF_LOW);
#define REGISTER_COPY_OPT_INSTANCE(id, name, Optimization)     \
  auto contig_opt_##id = std::make_unique<Optimization>();     \
  static register_opt::CopyOptBuilder register_contig_opt##id( \
      #name, contig_opt_##id);

}  // namespace sdaa_copy

}  // namespace custom_kernel
