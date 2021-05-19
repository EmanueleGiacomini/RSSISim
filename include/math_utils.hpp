/**
 * math_utils.hpp
 */

#pragma once

#include <cmath>

namespace rssisim {
    inline double wtdbm(double w) {return 10 * log10(w / 1e-3);} // Watt to dBm
    inline double dbmtw(double dbm) {return 1e-3 * pow(10, dbm/10);} // dBm to Watt
}