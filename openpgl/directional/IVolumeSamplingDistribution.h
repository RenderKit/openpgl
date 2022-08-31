// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../spatial/IRegion.h"

namespace openpgl
{
struct IVolumeSamplingDistribution
{
    //IVolumeSamplingDistribution() = delete;
    
    IVolumeSamplingDistribution() {};
    
    virtual ~IVolumeSamplingDistribution() {};
    
    virtual void init(const void* distribution, Point3 samplePosition) = 0;
    
    virtual Vector3 sample(const Point2 sample) const = 0;

    virtual float pdf(const Vector3 dir) const = 0;

    virtual float samplePdf(const Point2 sample, Vector3 &dir) const = 0;

    virtual bool validate() const = 0;

    virtual void clear() = 0;

    virtual void applySingleLobeHenyeyGreensteinProduct(const Vector3& dir, const float meanCosine) = 0;

    virtual bool supportsApplySingleLobeHenyeyGreensteinProduct() const = 0;

    virtual std::string toString() const = 0;

    virtual void setRegion(const IRegion* region) = 0;

    virtual const IRegion* getRegion() const = 0;

protected:
    //bool m_useParallaxCompensation{true};
    //const IRegion* m_region {nullptr};
};

}