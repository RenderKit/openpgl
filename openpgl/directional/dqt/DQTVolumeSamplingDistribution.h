#pragma once

#include "../IVolumeSamplingDistribution.h"
#include "DQT.h"

namespace openpgl
{

template<class TDirectionalQuadtree>
struct DQTVolumeSamplingDistribution: public IVolumeSamplingDistribution {
    DQTVolumeSamplingDistribution(/*const bool useParallaxCompensation):IVolumeSamplingDistribution(useParallaxCompensation*/)
    {};

    virtual ~DQTVolumeSamplingDistribution() override {};

    virtual void init(const void* distribution, Point3 samplePosition) override {
        this->distribution = *(TDirectionalQuadtree*)distribution;
    };

    inline Vector3 sample(const Point2 sample) const override {
        return distribution.sample(sample);
    };

    inline float pdf(const Vector3 dir) const override {
        return distribution.pdf(dir);
    };

    inline float samplePdf(const Point2 sample, Vector3 &dir) const override {
        return distribution.samplePdf(sample, dir);
    }

    inline bool validate() const override {
        return distribution.isValid();
    };

    inline void clear() override {
    };

    inline void applySingleLobeHenyeyGreensteinProduct(const Vector3& dir, const float meanCosine) override {
        // not supported by quadtree
        return;
    };

    inline bool supportsApplySingleLobeHenyeyGreensteinProduct() const override {
        return false;
    }

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