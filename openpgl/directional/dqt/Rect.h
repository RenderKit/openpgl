#pragma once

#include "../../openpgl_common.h"

namespace openpgl {

template <class T>
struct Rect {
    embree::Vec2<T> min;
    embree::Vec2<T> max;

    T area() {
        return (max.x - min.x) * (max.y - min.y);
    }

    Rect<T> intersect(const Rect<T> &other) const {
        Rect<T> isect;
        isect.min.x = std::max<T>(min.x, other.min.x);
        isect.min.y = std::max<T>(min.y, other.min.y);
        isect.max.x = std::min<T>(max.x, other.max.x);
        isect.max.y = std::min<T>(max.y, other.max.y);
        isect.max.x = std::max<T>(isect.min.x, isect.max.x);
        isect.max.y = std::max<T>(isect.min.y, isect.max.y);
        return isect;
    }

    Rect<T> child(uint32_t idx) const {
        const embree::Vec2<T> mid = (min + max) / (T)2;
        return {
            {
                (idx & 1) == 0 ? min.x : mid.x,
                (idx & 2) == 0 ? min.y : mid.y
            },
            {
                (idx & 1) == 0 ? mid.x : max.x,
                (idx & 2) == 0 ? mid.y : max.y
            }
        };
    }
};

}