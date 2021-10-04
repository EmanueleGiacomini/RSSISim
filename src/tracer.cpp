/**
 * tracer.cpp
 */

#include "tracer.hpp"

namespace rssisim {
    double raytrace_trace(const Map &world, const Eigen::Vector2f source, const Eigen::Vector2f target) {
        Eigen::Vector2f ray_dir = (target - source);
        ray_dir.normalize();
        rssisim::Ray2f ray(source, ray_dir);
        
        bool in_obstacle=false;
        rssisim::Obstacle *ref_obstacle=nullptr;
        std::shared_ptr<rssisim::PathLoss> it_loss=world.get_loss();

        double cumulative_loss = 0.0;
        
        // Casting loop
        while (!((target-ray.pos).norm() < 1e-3))  {
            if (in_obstacle) {
                it_loss = ref_obstacle->get_loss();
            } else {
                it_loss = world.get_loss();
            }

            double t = ray.cast(target);
            // AABB Intersection block
            double min_obst_t = INFINITY;
            rssisim::Obstacle *min_obst = nullptr;
            for (auto &obst : world.get_bbox_vect()) {
                double t_obst;
                bool obst_intersect = rssisim::intersect_bbox(ray, *obst, t_obst);
                //std::cout << "obstacle collision check=" << obst_intersect << " at distance=" << t_obst << std::endl;
                if (obst_intersect && t_obst < min_obst_t) {
                    min_obst_t = t_obst;
                    min_obst = obst;
                }
            }
            // check if we are intersecting an obstacle
            if (min_obst_t > 0 && min_obst_t < t && min_obst != nullptr) {
                // ray is colliding with an obstacle
                t = min_obst_t + 1e-3;
                in_obstacle=!in_obstacle;
                ref_obstacle = min_obst;
            }
            cumulative_loss += it_loss->sample(t);
            ray.pos += t * ray.dir;
        }
        return cumulative_loss;
    }
}
