#pragma once

#include "format/real_vector.hpp"
#include "hash/crosspolytope.hpp"
#include "hash/simhash.hpp"
#include "math.hpp"

#include <cmath>

namespace puffinn {
    struct L2Distance {
        using Format = RealVectorFormat;
        using DefaultHash = FHTCrossPolytopeHash;
        using DefaultSketch = SimHash;

        static float compute_similarity(float* lhs, float* rhs, unsigned int dimensions) {
            auto dist = l2_distance_float_sse(lhs, rhs, dimensions);
            // Convert to a similarity between 0 and 1,
            // which is needed to calculate collision probabilities.
            return 1.0/(dist+1.0);
        }
    };
}

