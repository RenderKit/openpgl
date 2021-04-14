// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "../include/openpgl/sampler.h"
namespace openpgl
{

struct Sampler
{

    virtual float next1D() = 0;

    virtual Point2 next2D() = 0;
};

struct SamplerC: public Sampler
{
    SamplerC(PGLSampler* sampler)
    {
        this->sampler =sampler;
    }

    float next1D()
    {
        return sampler->next1D(sampler->sampler);
    }

    Point2 next2D()
    {
        pgl_point2f sample2D = sampler->next2D(sampler->sampler);
        return Point2(sample2D.x, sample2D.y);
    }
private:
    PGLSampler* sampler;
};

}