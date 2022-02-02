#pragma once

#include "../../openpgl_common.h"

namespace openpgl {
    // We may remove this mapping at some point, because it has a singularity near z = 1
    class SphereToSquareLatLong {
    public:
        static Vector2 directionToPoint(const Vector3 &direction_) {
            auto direction = embree::clamp(direction_, Vector3(-1), Vector3(1));
            float theta = std::acos(direction.z);
            float phi = std::atan2(direction.y, direction.x);
            return embree::clamp(Vector2(theta / M_PI, phi / (2 * M_PI) + 0.5), Vector2(0), Vector2(1));
        }

        static Vector3 pointToDirection(const Vector2 &point_) {
            auto point = embree::clamp(point_, Vector2(0), Vector2(1));
            float theta = point.x * M_PI;
            float phi = (point.y - 0.5) * 2 * M_PI;
            return embree::clamp(Vector3(
                std::sin(theta) * std::cos(phi),
                std::sin(theta) * std::sin(phi),
                std::cos(theta)
            ), Vector3(-1), Vector3(1));
        }

        static float jacobian(const Vector2 &point_) {
            auto point = embree::clamp(point_, Vector2(0), Vector2(1));
            float theta = point.x * M_PI;
            return std::max(FLT_MIN, std::sin(theta) * 2.f * float(M_PI) * float(M_PI));
        }

        static float area(const Rect<float>& rect) {
            return 2 * M_PI * (rect.max.y - rect.min.y) * (std::cos(M_PI * rect.min.x) - std::cos(M_PI * rect.max.x));
        }
    };

    class SphereToSquareCylindrical {
    public:
        static Vector2 directionToPoint(const Vector3 &direction_) {
            auto direction = embree::clamp(direction_, Vector3(-1), Vector3(1));
            float cosTheta = direction.z;
            float phi = std::atan2(direction.y, direction.x);
            return embree::clamp(Vector2((cosTheta + 1) / 2, phi / (2 * M_PI) + 0.5), Vector2(0), Vector2(1));
        }

        static Vector3 pointToDirection(const Vector2 &point_) {
            auto point = embree::clamp(point_, Vector2(0), Vector2(1));
            float cosTheta = 2 * point.x - 1;
            float sinTheta = sqrt(1 - cosTheta * cosTheta);
            float phi = (point.y - 0.5) * 2 * M_PI;
            return embree::clamp(Vector3(
                sinTheta * std::cos(phi),
                sinTheta * std::sin(phi),
                cosTheta
            ), Vector3(-1), Vector3(1));
        }

        static float jacobian(const Vector2 &point_) {
            return 4.0f * (float)M_PI;
        }

        static float area(const Rect<float>& rect) {
            return 4.0f * (float)M_PI * (rect.max.x - rect.min.x) * (rect.max.y - rect.min.y);
        }
    };
}