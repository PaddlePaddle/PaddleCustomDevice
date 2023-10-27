// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <cnrt.h>

#include "runtime/runtime.h"

/*
 * CNRTEvents are movable not copyable wrappers around CNRT's events.
 */
struct CNRTEvent {
  // Constructors
  // Default value for "flags" is specified below - it's
  // CNRT_NOTIFIER_DISABLE_TIMING_ALL
  CNRTEvent() noexcept = default;
  explicit CNRTEvent(cnrtNotifierFlags flag, int dev_idx) noexcept
      : flag_{flag}, dev_idx_(dev_idx) {
    createEvent();
  }

  ~CNRTEvent() {
    try {
      if (is_created_) {
        PADDLE_ENFORCE_MLU_SUCCESS(cnrtNotifierDestroy(event_));
      }
    } catch (...) { /* No throw */
    }
  }

  CNRTEvent(const CNRTEvent&) = delete;
  CNRTEvent& operator=(const CNRTEvent&) = delete;

  CNRTEvent(CNRTEvent&& other) noexcept { moveHelper(std::move(other)); }
  CNRTEvent& operator=(CNRTEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator cnrtNotifier_t() const { return event(); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const CNRTEvent& left, const CNRTEvent& right) {
    return left.event_ < right.event_;
  }

  int device() const {
    if (is_created_) {
      return dev_idx_;
    } else {
      return -1;
    }
  }

  bool isCreated() const { return is_created_; }
  bool wasRecorded() const { return was_recorded_; }
  bool isCompleted() const { return is_completed_; }
  int device_index() const { return dev_idx_; }
  cnrtNotifier_t event() const { return event_; }

  bool query() {
    if (!is_created_) {
      return true;
    }

    cnrtRet_t ret_err = cnrtQueryNotifier(event_);
    if (ret_err == cnrtSuccess) {
      is_completed_ = true;
    } else {
      (void)cnrtGetLastError();
    }

    return is_completed_ ? true : false;
  }

  void record(const int device, cnrtQueue_t stream) {
    PADDLE_ENFORCE_EQ(
        dev_idx_,
        device,
        phi::errors::InvalidArgument(
            "Event device %d does not match recording stream's device %d.",
            dev_idx_,
            device));
    PADDLE_ENFORCE_MLU_SUCCESS(
        cnrtPlaceNotifier(reinterpret_cast<cnrtNotifier_t>(event_), stream));
    was_recorded_ = true;
  }

  void block(const cnrtQueue_t stream) {
    if (is_created_) {
      PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueWaitNotifier(
          reinterpret_cast<cnrtNotifier_t>(event_), stream, 0));
    }
  }

  void synchronize() const {
    if (is_created_) {
      PADDLE_ENFORCE_MLU_SUCCESS(
          cnrtWaitNotifier(reinterpret_cast<cnrtNotifier_t>(event_)));
    }
  }

 private:
  cnrtNotifierFlags flag_ =
      CNRT_NOTIFIER_DISABLE_TIMING_ALL;  // no time elapsing
  bool is_created_ = false;
  bool was_recorded_ = false;
  bool is_completed_ = false;
  int dev_idx_ = -1;
  cnrtNotifier_t event_ = nullptr;

  void createEvent() {
    // dev_idx_ = dev_idx;
    PADDLE_ENFORCE_MLU_SUCCESS(cnrtNotifierCreateWithFlags(&event_, flag_));
    is_created_ = true;
  }

  void moveHelper(CNRTEvent&& other) {
    std::swap(flag_, other.flag_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(dev_idx_, other.dev_idx_);
    std::swap(event_, other.event_);
  }
};
