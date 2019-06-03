#pragma once

#include "format/set.hpp"
#include "hash/minhash.hpp"

namespace puffinn {
    struct JaccardSimilarity {
        using Format = SetFormat; 
        using DefaultSketch = MinHash1Bit;
        using DefaultHash = MinHash;

        static float compute_similarity(Format::Type* lhs_ptr, Format::Type* rhs_ptr, unsigned int) {
            auto& lhs = *lhs_ptr;
            auto& rhs = *rhs_ptr;
            int intersection_size = 0;
            size_t lhs_idx = 0;
            size_t rhs_idx = 0;
            while (lhs_idx < lhs.size() && rhs_idx < rhs.size()) {
                if (lhs[lhs_idx] == rhs[rhs_idx]) {
                    intersection_size++;
                    lhs_idx++;
                    rhs_idx++;
                } else if(lhs[lhs_idx] < rhs[rhs_idx]) {
                    lhs_idx++;
                } else {
                    rhs_idx++;
                }
            }
            float intersection = intersection_size;
            auto divisor = lhs.size()+rhs.size()-intersection;
            if (divisor == 0) {
                return 0;
            } else {
                return intersection/divisor;
            }
        }
    };
}
