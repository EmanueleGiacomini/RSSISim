/**
 * aabb.hpp
 */

#pragma once

#include "obstacle.hpp"
#include "path_loss/path_loss.hpp"
#include <Eigen/Dense>

namespace rssisim {
    class AxisAlignedBoundingBox {
        PathLoss *obst_loss;
        public:
        Eigen::Vector2f vert_min;
        Eigen::Vector2f vert_max;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        AxisAlignedBoundingBox(Eigen::Vector2f _vert_min, Eigen::Vector2f _vert_max, PathLoss *_obst_loss): vert_min(_vert_min), vert_max(_vert_max), obst_loss(_obst_loss) {}
        AxisAlignedBoundingBox(Eigen::Vector2f _vert_min, Eigen::Vector2f _vert_max) : vert_min(_vert_min), vert_max(_vert_max), obst_loss(nullptr) {}
        bool intersect(const Eigen::Vector2f &p, const Eigen::Vector2f &d, Eigen::Vector2f &p_start, Eigen::Vector2f &p_end);
        double sample(double distance);
        inline void set_loss(PathLoss *_obst_loss) {obst_loss = _obst_loss;}
    };
}