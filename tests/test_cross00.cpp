/**
 * test_cross00.cpp
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <memory>
#include <cstdlib>
#include <ctime>

#include "obstacle/aabb.hpp"
#include "path_loss/isotropic_ploss.hpp"
#include "path_loss/ideal_ploss.hpp"
#include "intersection.hpp"
#include "ray.hpp"
#include "math_utils.hpp" 

#define NUM_SAMPLES 500

double raytrace_trace(const rssisim::Map &world, const Eigen::Vector2f source, const Eigen::Vector2f target) {
    Eigen::Vector2f ray_dir = (target - source);
    ray_dir.normalize();
    rssisim::Ray2f ray(source, ray_dir);
    
    bool in_obstacle=false;
    rssisim::Obstacle *ref_obstacle=nullptr;
    std::shared_ptr<rssisim::PathLoss> it_loss=world.get_loss();

    double cumulative_loss = 0.0;
    std::size_t russian_roulette = 0;
    // Casting loop
    while (!((target-ray.pos).norm() < 1e-3) && russian_roulette < 20)  {
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
        russian_roulette++;
    }
    return cumulative_loss;
}

const double TX_POWER = 1; // [W]
const double BLE_LAMBDA = 0.1206; // BLE wavelength [m]

int main(int argc, char **argv) {
    std::shared_ptr<rssisim::IdealPathLoss> world_loss(new rssisim::IdealPathLoss(BLE_LAMBDA));
    std::shared_ptr<rssisim::IsotropicPathLoss> concrete_ploss(new rssisim::IsotropicPathLoss(0.001));
    // Generate world and register some obstacles
    rssisim::Map world(world_loss);
    // 26.8, 1.5 -> 38, 24
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(27, 2), Eigen::Vector2f(38, 24), concrete_ploss));
    // 60, 3 -> 73, 10
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(60, 3), Eigen::Vector2f(73, 10), concrete_ploss));
    // 2, 36 -> 22, 41
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(2, 36), Eigen::Vector2f(22, 41), concrete_ploss));
    // 87,16 -> 98, 34
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(87, 16), Eigen::Vector2f(98, 34), concrete_ploss));
    // 82,43 -> 98, 50
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(82, 43), Eigen::Vector2f(98, 50), concrete_ploss));
    // 23,52 -> 36, 62
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(23, 52), Eigen::Vector2f(36, 62), concrete_ploss));
    // 65,63 -> 74, 78
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(65, 63), Eigen::Vector2f(74, 78), concrete_ploss));
    // 3,68 -> 24, 78
    world.add(new rssisim::AxisAlignedBoundingBox(Eigen::Vector2f(3, 68), Eigen::Vector2f(24, 78), concrete_ploss));
    
    // Gw0 : 6, 25
    // Gw1 : 86, 10
    // Gw2 : 84, 60
    Eigen::Vector2f gw0(6, 25);
    Eigen::Vector2f gw1(86, 10);
    Eigen::Vector2f gw2(84, 60);
    // Initialize seed
    // Map has bounds (-1000, -1000), (1000, 1000)
    std::srand(41);
    // Store results on file
    std::ofstream file;
    file.open("cross01.csv");
    file << "x,y,recv_pwr0,recv_pwr1,recv_pwr2\n";
    for(std::size_t i=0; i < NUM_SAMPLES; ++i) {
        // Generate target
        Eigen::Vector2f target = 100 * (Eigen::Vector2f(1, 1) + Eigen::Vector2f::Random(2)) / 2;
        double recv_loss0 = raytrace_trace(world, gw0, target);
        double recv_loss1 = raytrace_trace(world, gw1, target);
        double recv_loss2 = raytrace_trace(world, gw2, target);
        double recv_pwr0 = rssisim::wtdbm(TX_POWER) + recv_loss0;
        double recv_pwr1 = rssisim::wtdbm(TX_POWER) + recv_loss1;
        double recv_pwr2 = rssisim::wtdbm(TX_POWER) + recv_loss2;
        std::cout << i << std::endl;
        file << target.x() << "," << target.y() << "," << recv_pwr0 << "," << recv_pwr1 << "," << recv_pwr2 << "\n"; 
    }

    std::cout << "Done" << std::endl;
    file.close();
    return 0;
}