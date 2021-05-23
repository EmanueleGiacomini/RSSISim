/**
 * obstacle.hpp
 */

#pragma once
#include <Eigen/Dense>
#include <path_loss/path_loss.hpp>

namespace rssisim {
    class Obstacle {
        protected:
        PathLoss *p_loss;
        Obstacle(PathLoss *_p_loss) : p_loss(_p_loss) {}
        public:
        inline void add_loss(PathLoss *_p_loss) {p_loss = _p_loss;}
        inline PathLoss *get_loss() {return p_loss;}
    };
}