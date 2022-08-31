#pragma once

#include "../ISurfaceSamplingDistribution.h"
#include "DQT.h"

namespace openpgl
{

template<class TDirectionalQuadtree>
struct DQTSurfaceSamplingDistribution: public ISurfaceSamplingDistribution {
    DQTSurfaceSamplingDistribution(/*const bool useParallaxCompensation):ISurfaceSamplingDistribution(useParallaxCompensation*/)
    {};

    virtual ~DQTSurfaceSamplingDistribution() override {};

    virtual void init(const void* distribution, Point3 samplePosition) override {
        this->distribution = *(TDirectionalQuadtree*)distribution;
    };

    inline void applyCosineProduct(const Vector3& normal) override {
        // not supported by quadtree
        return;
    };

    inline bool supportsApplyCosineProduct() const override {
        return false;
    }

    inline Vector3 sample(const Point2 sample) const override {
        return distribution.sample(sample);
    };

    inline float pdf(const Vector3 dir) const override {
        return distribution.pdf(dir);
    };

    inline float samplePdf(const Point2 sample, Vector3 &dir) const override
    {
        return distribution.samplePdf(sample, dir);
    }

    inline bool validate() const override {
        return distribution.isValid();
    };

    inline void clear() override {
    };

    std::string toString() const override {
        return "";
    };

    const IRegion* getRegion() const override {
        return m_region;
    }

    void setRegion(const IRegion* region) override {
        m_region = region;
    }

private:
    TDirectionalQuadtree distribution;
    const IRegion* m_region {nullptr};
};

}