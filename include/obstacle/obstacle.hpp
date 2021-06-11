/**
 * obstacle.hpp
 */

#pragma once
#include <Eigen/Dense>
#include <memory>
#include <path_loss/path_loss.hpp>

namespace rssisim {
    class Obstacle {
        protected:
        std::shared_ptr<PathLoss> p_loss;
        Obstacle(PathLoss *_p_loss) : p_loss(_p_loss) {}
        Obstacle(std::shared_ptr<PathLoss> _p_loss) : p_loss(_p_loss) {}
        public:
        inline void add_loss(std::shared_ptr<PathLoss> _p_loss) {p_loss = _p_loss;}
        inline std::shared_ptr<PathLoss> get_loss() const {return p_loss;}
    };
}