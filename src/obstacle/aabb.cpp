/**
 * aabb.cpp
 */

#include "obstacle/aabb.hpp"
#include <Eigen/Dense>

namespace rssisim {
    bool AxisAlignedBoundingBox::intersect(const Eigen::Vector2f &p, const Eigen::Vector2f &d, Eigen::Vector2f &p_start, Eigen::Vector2f &p_end) {
        return false;
    }

    double AxisAlignedBoundingBox::sample(double distance) {
        return 0.0;
    }
}