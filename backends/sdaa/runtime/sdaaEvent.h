// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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

#include <utility>

#include "runtime/runtime.h"

/*
 * sdaaEvents are movable not copyable wrappers around sdaa's events.
 */
struct sdaaEventWrapper {
  // Constructors
  sdaaEventWrapper() noexcept = default;
  explicit sdaaEventWrapper(int dev_idx) noexcept : dev_idx_(dev_idx) {
    createEvent();
  }

  ~sdaaEventWrapper() {
    try {
      if (is_created_) {
        synchronize();
        checkSdaaErrors(sdaaEventDestroy(event_));
      }
    } catch (...) { /* No throw */
    }
  }

  sdaaEventWrapper(const sdaaEventWrapper&) = delete;
  sdaaEventWrapper& operator=(const sdaaEventWrapper&) = delete;

  sdaaEventWrapper(sdaaEventWrapper&& other) noexcept {
    moveHelper(std::move(other));
  }

  sdaaEventWrapper& operator=(sdaaEventWrapper&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator sdaaEvent_t() const { return event(); }

  // Less than operator is not supported at present

  int device() const {
    if (is_created_) {
      return dev_idx_;
    } else {
      return -1;
    }
  }

  bool isCreated() const { return is_created_; }
  bool wasRecorded() const { return was_recorded_; }
  int device_index() const { return dev_idx_; }
  sdaaEvent_t event() const { return event_; }

  bool query() {
    if (!is_created_) {
      return true;
    }

    auto ret_err = sdaaEventQuery(event_);
    if (ret_err == sdaaSuccess) {
      return true;
    } else if (ret_err != sdaaErrorNotReady) {
      checkSdaaErrors(ret_err);
    } else {
      // ignore and clear the error if not ready
      (void)sdaaGetLastError();
    }

    return false;
  }

  void record(const int device, sdaaStream_t stream) {
    if (!is_created_) {
      createEvent();
    }

    PADDLE_ENFORCE_EQ(
        dev_idx_,
        device,
        phi::errors::InvalidArgument(
            "Event device %d does not match recording stream's device %d.",
            dev_idx_,
            device));
    checkSdaaErrors(sdaaEventRecord(event_, stream));
    was_recorded_ = true;
  }

  void block(const sdaaStream_t stream) {
    if (is_created_) {
      checkSdaaErrors(sdaaStreamWaitEvent(stream, event_, 0));
    }
  }

  void synchronize() const {
    if (is_created_) {
      checkSdaaErrors(sdaaEventSynchronize(event_));
    }
  }

 private:
  bool is_created_ = false;
  bool was_recorded_ = false;
  int dev_idx_ = -1;
  sdaaEvent_t event_ = nullptr;

  void createEvent() {
    // TODO(teco): create event with sdaaEventCreateWithFlags when it
    // supports set flags
    checkSdaaErrors(sdaaEventCreate(&event_));
    is_created_ = true;
  }

  void moveHelper(sdaaEventWrapper&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(dev_idx_, other.dev_idx_);
    std::swap(event_, other.event_);
  }
};
