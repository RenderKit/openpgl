#pragma once

#include <glm/vec3.hpp>
#include <algorithm>
#include <limits>

struct AABB{

    AABB(){
        min = glm::vec3(std::numeric_limits<float>::max());
        max = glm::vec3(std::numeric_limits<float>::min());
    }

    glm::vec3 min;
    glm::vec3 max;

    glm::vec3 getExtend() const {
        return max-min;
    }

    void extend(const glm::vec3 v)
    {
        min[0] = std::min(min[0], v[0]);
        min[1] = std::min(min[1], v[1]);
        min[2] = std::min(min[2], v[2]);

        max[0] = std::max(max[0], v[0]);
        max[1] = std::max(max[1], v[1]);
        max[2] = std::max(max[2], v[2]);
    }

};