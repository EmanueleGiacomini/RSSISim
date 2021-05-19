/**
 * ideal_ploss.hpp
 */

#include "path_loss.hpp"
#include <cmath>

namespace rssisim {
    class IdealPathLoss: public PathLoss {
        double lambda; // Wavelenght of the signal [m]
        public:
        IdealPathLoss(double _lambda): lambda(_lambda) {}
        inline double sample(double distance) {return -20 * log10(4 * M_PI * distance / lambda);}
    };
}