#pragma once
#include <iostream>
#include "glog/logging.h"

class hpu_logging {
  public:
    hpu_logging(std::string func) { function = func; LOG(INFO) << "Enter " << function << std::endl;};
    ~hpu_logging() { LOG(INFO) << "Exit " << function << std::endl;};
  private:
    std::string function;
};

#define HPU_LOG hpu_logging local_log = hpu_logging(__FUNCTION__);

#define  FENTRY() (LOG(INFO) << "Enter " << __FUNCTION__ << std::endl);
#define  FEXIT() (LOG(INFO) << "Exit " << __FUNCTION__ << std::endl);
