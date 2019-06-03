#pragma once

#include "format/unit_vector.hpp"
#include "hash/simhash.hpp"
#include "hash/crosspolytope.hpp"
#include "math.hpp"

namespace puffinn {
    struct CosineSimilarity {
        using Format = UnitVectorFormat;
        using DefaultHash = FHTCrossPolytopeHash;
        using DefaultSketch = SimHash;

        static float compute_similarity(int16_t* lhs, int16_t* rhs, unsigned int dimensions) {
            float dot = Format::from_16bit_fixed_point(
                dot_product_i16_avx2(lhs, rhs, dimensions));
            return (dot+1)/2; // Ensure the similarity is between 0 and 1.
        }
    };
}
