#pragma once

#include <murand.h>
#include <thrust/system/musa/error.h>
#include <thrust/system_error.h>

using curandStatus_t = murandStatus_t;

using curandCreateGenerator = murandCreateGenerator;
using curandSetStream = murandSetStream;
using curandSetPseudoRandomGeneratorSeed = murandSetPseudoRandomGeneratorSeed;
using curandGenerateUniform = murandGenerateUniform;
using curandGenerateUniformDouble = murandGenerateUniformDouble;
using curandGenerateNormal = murandGenerateNormal;
using curandDestroyGenerator = murandDestroyGenerator;

#define CURAND_STATUS_SUCCESS MURAND_STATUS_SUCCESS