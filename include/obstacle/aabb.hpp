/**
 * aabb.hpp
 */

#pragma once

#include "obstacle.hpp"
#include "path_loss/path_loss.hpp"
#include <Eigen/Dense>

namespace rssisim {
    class AxisAlignedBoundingBox: public ObstacleInterface {
        Eigen::Vector2f vert_a;
        Eigen::Vector2f vert_b;
        PathLoss &obst_loss;
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        AxisAlignedBoundingBox(Eigen::Vector2f &_vert_a, Eigen::Vector2f &_vert_b, PathLoss &_obst_loss): vert_a(_vert_a), vert_b(_vert_b), obst_loss(_obst_loss) {}
        bool intersect(const Eigen::Vector2f &p, const Eigen::Vector2f &d, Eigen::Vector2f &p_start, Eigen::Vector2f &p_end);
        double sample(double distance);
    };
}