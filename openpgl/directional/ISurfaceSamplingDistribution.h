// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace openpgl
{
struct ISurfaceSamplingDistribution
{
    virtual ~ISurfaceSamplingDistribution() {};
    
    virtual void init(const void* distribution) = 0;
    
    virtual void applyCosineProduct(const Vector3& normal) = 0;
    
    virtual Vector3 sample(const Point2 sample) const = 0;

    virtual float pdf(const Vector3 dir) const = 0;

    virtual bool valid() const = 0;

    virtual void clear() = 0;

    virtual std::string toString() const = 0;
};
}