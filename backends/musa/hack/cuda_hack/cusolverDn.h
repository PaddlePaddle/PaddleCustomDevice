#pragma once

#include <musolver.h>

typedef enum fakeMusolverStatus_t {
    CUSOLVER_STATUS_SUCCESS         = 0, /**< success */
} musolverStatus;


using cusolverStatus_t = fakeMusolverStatus_t;