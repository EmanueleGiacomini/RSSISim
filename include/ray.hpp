/*
 * ray.hpp
 */

#pragma once
#include <Eigen/Dense>
#include "map.hpp"
// To be removed
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

  struct Ray2f{
    Eigen::Vector2f pos;
    Eigen::Vector2f dir;
    Eigen::Vector2f idir;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Ray2f() {};
    Ray2f(Eigen::Vector2f _pos, Eigen::Vector2f _dir): pos(_pos), dir(_dir) {
      idir.x() = 1/dir.x();
      idir.y() = 1/dir.y();
    }
    
    inline double cast(Eigen::Vector2f target) {return (target.x() - pos.x()) / dir.x();}
  };
}
