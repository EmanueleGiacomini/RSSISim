/**
 * aabb.hpp
 */

#pragma once

#include "obstacle.hpp"
#include "path_loss/path_loss.hpp"
#include <Eigen/Dense>

namespace rssisim {
    class AxisAlignedBoundingBox : public Obstacle {
        public:
        Eigen::Vector2f vert_min;
        Eigen::Vector2f vert_max;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        AxisAlignedBoundingBox(Eigen::Vector2f _vert_min, Eigen::Vector2f _vert_max, PathLoss *_obst_loss): vert_min(_vert_min), vert_max(_vert_max), Obstacle(_obst_loss) {}
        AxisAlignedBoundingBox(Eigen::Vector2f _vert_min, Eigen::Vector2f _vert_max) : vert_min(_vert_min), vert_max(_vert_max), Obstacle(nullptr) {}
        double sample(double distance);
    };
}