/**
 * test_intersection.cpp
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <Eigen/Dense>

#include "obstacle/aabb.hpp"
#include "path_loss/isotropic_ploss.hpp"
#include "path_loss/ideal_ploss.hpp"
#include "intersection.hpp"
#include "ray.hpp"
#include "math_utils.hpp"

double raytrace_trace(const rssisim::Map &world, const Eigen::Vector2f source, const Eigen::Vector2f target) {
    Eigen::Vector2f ray_dir = (target - source);
    ray_dir.normalize();
    rssisim::Ray2f ray(source, ray_dir);
    
    bool in_obstacle=false;
    rssisim::Obstacle *ref_obstacle=nullptr;
    rssisim::PathLoss *it_loss=world.get_loss();

    double cumulative_loss = 0.0;

    // Casting loop
    while (!((ray.pos-target).norm() < 1e-6)) {

        if (in_obstacle) {
            it_loss = ref_obstacle->get_loss();
        } else {
            it_loss = world.get_loss();
        }

        double t = ray.cast(target);
        // AABB Intersection block
        double min_obst_t = 1e9;
        rssisim::Obstacle *min_obst = nullptr;
        std::vector<rssisim::AxisAlignedBoundingBox> aabb_vect=world.get_bbox_vect();
        for (auto &obst : aabb_vect) {
            double t_obst;
            bool obst_intersect = rssisim::intersect_bbox(ray, obst, t_obst);
            //std::cout << t_obst << ", " << obst_intersect << std::endl;
            if (obst_intersect && t_obst < min_obst_t) {
                min_obst_t = t_obst;
                min_obst = &obst;
            }
        }
        // check if we are intersecting an obstacle
        if (min_obst_t < t && min_obst != nullptr) {
            // ray is colliding with an obstacle
            t = min_obst_t + 1e-9;
            in_obstacle=!in_obstacle;
            ref_obstacle = min_obst;
            //std::cout << "Colliding with a box at distance t=" << t <<", entering=" << in_obstacle << std::endl;
        }
        
        cumulative_loss += it_loss->sample(t);
        ray.pos += t * ray.dir;

    }
    return cumulative_loss;
}

const double TX_POWER = 1; // [W]
const double BLE_LAMBDA = 0.1206; // BLE wavelenght [m]

int main(int argc, char **argv) {
    rssisim::IdealPathLoss world_loss(BLE_LAMBDA);
    rssisim::IsotropicPathLoss concrete_ploss(0.5);
    // Generate world and register some obstacles
    rssisim::Map world(&world_loss);
    world.add(rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(0.5, -0.5), Eigen::Vector2f(1.5, 0.5), &concrete_ploss));
    world.add(rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(1, 1),      Eigen::Vector2f(3, 2), &concrete_ploss));
    world.add(rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(2, -1),     Eigen::Vector2f(5, 1.5), &concrete_ploss));
    world.add(rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(-5, -1),    Eigen::Vector2f(3, 3), &concrete_ploss));


    // Store results on file
    std::ofstream file;
    file.open("data.csv");
    file << "x,y,recv_pwr\n";

    for (double x=0.5; x < 100; x+=0.1) {
        for (double y=0.5; y < 100; y+=0.1) {
            Eigen::Vector2f target(x, y);
            double recv_loss = raytrace_trace(world, Eigen::Vector2f(0, 0), target);
            std::cout << recv_loss << std::endl;
            double recv_pwr = rssisim::dbmtw(rssisim::wtdbm(TX_POWER) + recv_loss);

            //file << x << "," << y << "," << recv_pwr << "\n";
        }
    }
    file.close();
    return 0;
}