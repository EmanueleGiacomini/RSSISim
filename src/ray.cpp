/**
 * ray.cpp
 */

#include <iostream>
#include <cmath>
#include "ray.hpp"
#include "math_utils.hpp"

namespace rssisim {

    double Ray::cast(Eigen::Vector2f target) {
        double ray_pwr = wtdbm(tx_power);
        // Compute ray direction
        Eigen::Vector2f dir = (target - pos);
        dir.normalize();
        // Tangential distance from source to target
        double target_t = - (target[0] - pos[0] + target[1] - pos[1]) / (dir[0] + dir[1]);
        std::cout << pos.transpose() << ", " << dir.transpose() << ", " << target_t << std::endl;
        // Apply free space path loss
        double path_loss = fs_loss.sample(target_t);
        ray_pwr += path_loss;
        return ray_pwr;
    }

}