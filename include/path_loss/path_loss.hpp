/**
 * path_loss.hpp
 */

#pragma once

namespace rssisim {
    class PathLoss {
        public:
        virtual double sample(double distance)=0; // Returns the path loss in dB
    };
}