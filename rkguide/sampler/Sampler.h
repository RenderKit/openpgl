// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

namespace rkguide
{

struct Sampler
{

    virtual float next1D() = 0;

    virtual Point2 next2D() = 0;
};

}