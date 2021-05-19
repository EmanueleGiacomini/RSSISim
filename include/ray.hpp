/*
 * ray.hpp
 */

#pragma once
#include <Eigen/Dense>
#include "path_loss/path_loss.hpp"

namespace rssisim {
  class Ray {
    /**
     * TODO DOCS
     */
    Eigen::Vector2f pos;
    double tx_power;
    PathLoss &fs_loss; // Free-Space Loss
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Ray(Eigen::Vector2f _pos, double _tx_power, PathLoss &_fs_loss) : pos(_pos), tx_power(_tx_power), fs_loss(_fs_loss) {};
    double cast(Eigen::Vector2f target);
  };
}
