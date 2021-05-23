/**
 * map.hpp
 */

#pragma once
#include "obstacle/obstacle.hpp"
#include "obstacle/aabb.hpp"
#include "path_loss/path_loss.hpp"
#include <vector>

namespace rssisim {
    class Map {
        PathLoss *p_loss;
        std::vector<AxisAlignedBoundingBox> bbox_vect;
        public:
        Map(PathLoss *_p_loss) : p_loss(_p_loss) {};
        inline void add(AxisAlignedBoundingBox bbox) {bbox_vect.push_back(bbox);}
        inline std::vector<AxisAlignedBoundingBox> get_bbox_vect() const {return bbox_vect;}
        inline PathLoss *get_loss() const {return p_loss;}
    };
}