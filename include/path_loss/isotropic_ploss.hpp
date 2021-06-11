/**
 * isotropic_ploss.hpp
 */

#include "path_loss/path_loss.hpp"

namespace rssisim {
    class IsotropicPathLoss: public PathLoss {
        double attenuation; // Attenuation in dB/mm
        public:
        IsotropicPathLoss(double _attenuation) : attenuation(_attenuation) {}
        inline double sample(double distance) {return - 1e3 * distance * attenuation;} //assuming distance is in meters
    };
}