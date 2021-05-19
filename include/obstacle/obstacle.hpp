/**
 * obstacle.hpp
 */

#pragma once
#include <Eigen/Dense>

namespace rssisim {
    class ObstacleInterface {
        public:
        virtual bool intersect(const Eigen::Vector2f &p, const Eigen::Vector2f &d, Eigen::Vector2f &p_start, Eigen::Vector2f &p_end)=0;
        virtual double sample(double distance)=0;
    };
}