/**
 * intersection.hpp
 */

#pragma once

#include "ray.hpp"
#include "obstacle/aabb.hpp"
#include <algorithm> // min/max

namespace rssisim {
    inline bool intersect_bbox(const Ray2f &ray, const AxisAlignedBoundingBox &bbox, double &t) {
        // Check for Ray2f-AxisAlignedBoundingBox intersection
        // https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
        Ray2f idir;
        idir.dir.x() = 1.0 / ray.dir.x();
        idir.dir.y() = 1.0 / ray.dir.y();

        double t1 = (bbox.vert_min.x() - ray.pos.x()) * idir.dir.x();
        double t2 = (bbox.vert_max.x() - ray.pos.x()) * idir.dir.x();
        double t3 = (bbox.vert_min.y() - ray.pos.y()) * idir.dir.y();
        double t4 = (bbox.vert_max.y() - ray.pos.y()) * idir.dir.y();

        double t_min = std::max(std::min(t1, t2), std::min(t3, t4));
        double t_max = std::min(std::max(t1, t2), std::max(t3, t4));

        // if t_max < 0 the ray is intersecting AABB, but the whole AABB is behind it
        if (t_max < 0) {
            t = t_max;
            return false;
        }
        // if t_min > t_max, ray doesn't intersect AABB
        if (t_min > t_max) {
            t = t_max;
            return false;
        }
        t = t_min;
        return true;
    }
    // TODO (Add other obstacle intersection functions)
}

