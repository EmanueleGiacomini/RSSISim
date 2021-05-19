/**
 * test_raycasting.cpp
 */

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "ray.hpp"
#include "path_loss/ideal_ploss.hpp"
#include "math_utils.hpp"

const double TX_POWER = 1; // [W]
const double BLE_LAMBDA = 0.1206; // BLE wavelenght [m]

int main(int argc, char **argv) {
    Eigen::RowVector2f ray_orig(0, 0);
    rssisim::IdealPathLoss fs_loss(BLE_LAMBDA);
    rssisim::Ray ray(ray_orig, TX_POWER, fs_loss);

    // Store results on file
    std::ofstream file;
    file.open("data.csv");
    file << "x,y,recv_pwr\n";
    for (double x=0.5; x < 100; x+=0.1) {
        for (double y=0.5; y < 100; y+=0.1) {
            Eigen::RowVector2f target(x, y);
            double recv_pwr = ray.cast(target);
            file << x << "," << y << "," << recv_pwr << "\n";
        }
    }
    file.close();
    return 0;
}