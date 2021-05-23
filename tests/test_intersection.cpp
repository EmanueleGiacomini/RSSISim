/**
 * test_intersection.cpp
 */

#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include "obstacle/aabb.hpp"
#include "intersection.hpp"
#include "ray.hpp"


int main(int argc, char **argv) {
    // Ray starting from origin and following X axis
    rssisim::Ray2f ray(Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 0));
    // AABB which lies on the X axis
    rssisim::AxisAlignedBoundingBox obstacle(Eigen::Vector2f(0.5, -0.5), Eigen::Vector2f(1.5, 0.5));
    Eigen::Vector2f target(2, 0);
    while (ray.pos != target) {
        double t = ray.cast(target);
        // Check for intersections
        double t_obst;
        bool intersect_flag = rssisim::intersect_bbox(ray, obstacle, t_obst);
        
        // If intersection happens, reach the obstacle
        if (intersect_flag && t_obst > 0.0) {
            if (t > t_obst) {
                // collision before reaching the target
                t = t_obst + 1e-6; // adding 1e-6 to avoid new collision
                std::cout << "Collision at " << t << std::endl;
            }
        }            
        ray.pos += ray.dir * t;
        std::cout << "Ray pos: " << ray.pos << std::endl;

    }

    return 0;
}