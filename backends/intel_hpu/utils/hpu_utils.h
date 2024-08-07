#pragma once
#include <iostream>

#include "glog/logging.h"

// #define FUNCALL_LOG hpu_logging local_log = hpu_logging(__FUNCTION__);
// function entry log
#define FUNCALL_S \
  { LOG(INFO) << "Enter " << __FUNCTION__ << std::endl; };
// function return log
#define FUNCALL_E \
  { LOG(INFO) << "Exit  " << __FUNCTION__ << std::endl; };

