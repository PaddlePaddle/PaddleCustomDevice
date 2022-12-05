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

// #define FUNCALL_LOG hpu_logging local_log = hpu_logging(__FUNCTION__);
//function entry log
#define FUNCALL_S {LOG(INFO) << "Enter " << __FUNCTION__ << std::endl;};
//function return log
#define FUNCALL_E {LOG(INFO) << "Exit  " << __FUNCTION__ << std::endl;};

#define CHKSTATUS(errstr) assert(status == synSuccess && errstr)