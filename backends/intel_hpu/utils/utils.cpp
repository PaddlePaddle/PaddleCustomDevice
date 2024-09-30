
#include "utills.h"

std::string ShowErrorMsg(synStatus s) {
  char msg[STATUS_DESCRIPTION_MAX_SIZE] = {0};

  synStatusGetBriefDescription(s, msg, STATUS_DESCRIPTION_MAX_SIZE);
  return std::string(msg);
}