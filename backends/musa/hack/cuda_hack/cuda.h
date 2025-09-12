#pragma once

#include <musa.h>
#include <driver_types.h>

using CUresult = MUresult;
using cudaError_t = musaError_t;
using cudaError_enum = musaError_enum;

#define cudaSuccess musaSuccess
#define CUDA_SUCCESS MUSA_SUCCESS 