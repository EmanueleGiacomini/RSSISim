/**
 * tracer.hpp
 */

#include "obstacle/aabb.hpp"
#include "map.hpp"
#include "intersection.hpp"
#include "ray.hpp"

#include <Eigen/Dense>

namespace rssisim {
    double raytrace_trace(const Map &world, const Eigen::Vector2f source, const Eigen::Vector2f target);
}
