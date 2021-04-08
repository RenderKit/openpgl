// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
struct Sampler
{
};
#else
typedef ManagedObject Region;
#endif

typedef Sampler *PGLSampler;




#ifdef __cplusplus
}  // extern "C"
#endif