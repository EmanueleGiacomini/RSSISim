/**
 * map.hpp
 */

#pragma once
#include "obstacle/obstacle.hpp"
#include "obstacle/aabb.hpp"
#include <vector>

namespace rssisim {
    class Map {
        std::vector<AxisAlignedBoundingBox> bbox_vect;
        public:
        Map();
        inline void add(AxisAlignedBoundingBox bbox) {bbox_vect.push_back(bbox);}
    };
}