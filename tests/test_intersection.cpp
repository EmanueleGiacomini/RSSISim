/**
 * test_intersection.cpp
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <Eigen/Dense>
#include <memory>

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
    std::shared_ptr<rssisim::PathLoss> it_loss=world.get_loss();

    double cumulative_loss = 0.0;

    // Casting loop
    while (!((target-ray.pos).norm() < 1e-3)) {
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

const double TX_POWER = 1; // [W]
const double BLE_LAMBDA = 0.1206; // BLE wavelenght [m]

int main(int argc, char **argv) {
    std::shared_ptr<rssisim::IdealPathLoss> world_loss(new rssisim::IdealPathLoss(BLE_LAMBDA));
    std::shared_ptr<rssisim::IsotropicPathLoss> concrete_ploss(new rssisim::IsotropicPathLoss(0.01));
    // Generate world and register some obstacles
    rssisim::Map world(world_loss);
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(50, 40), Eigen::Vector2f(80, 41), concrete_ploss));
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(20, 0), Eigen::Vector2f(40, 4), concrete_ploss));
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(0, 70), Eigen::Vector2f(40, 75), concrete_ploss));
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(30, 0), Eigen::Vector2f(50, 10), concrete_ploss));


    // Store results on file
    std::ofstream file;
    file.open("data.csv");
    file << "x,y,recv_pwr0,recv_pwr1,recv_pwr2\n";
    
    for (double x=0.5; x < 100; x+=1) {
        for (double y=0.5; y < 100; y+=1) {   
            Eigen::Vector2f target(x, y);
            double recv_loss0 = raytrace_trace(world, Eigen::Vector2f(0, 0), target);
            double recv_loss1 = raytrace_trace(world, Eigen::Vector2f(80, 101), target);
            double recv_loss2 = raytrace_trace(world, Eigen::Vector2f(101, 0), target);
            //std::cout << recv_loss << std::endl;
            double recv_pwr0 = rssisim::wtdbm(TX_POWER) + recv_loss0;
            double recv_pwr1 = rssisim::wtdbm(TX_POWER) + recv_loss1;
            double recv_pwr2 = rssisim::wtdbm(TX_POWER) + recv_loss2;

            file << x << "," << y << "," << recv_pwr0 << "," << recv_pwr1 << "," << recv_pwr2 << "\n";
        }
    }
    
    std::cout << "Done" << std::endl;
    //file.close();
    return 0;
}