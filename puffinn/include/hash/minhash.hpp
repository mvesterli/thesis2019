#pragma once

#include "format/set.hpp"
#include <random>

namespace puffinn {
    class TabulationHash {
        uint64_t t1[256];
        uint64_t t2[256];
        uint64_t t3[256];
        uint64_t t4[256];

    public:
        TabulationHash(std::mt19937_64& rng) {
            for (size_t i=0; i < 256; i++) {
                t1[i] = rng();
                t2[i] = rng();
                t3[i] = rng();
                t4[i] = rng();
            }
        }

        uint64_t operator()(uint32_t val) const {
            return (
                t1[val & 0xFF] ^
                t2[(val >> 8) & 0xFF] ^
                t3[(val >> 16) & 0xFF] ^
                t4[(val >> 24) & 0xFF]);
        }
    };

    struct MinHashArgs {};

    class MinHashFunction {
        TabulationHash hash;

    public:
        MinHashFunction(TabulationHash hash) : hash(hash) {
        }

        LshDatatype operator()(std::vector<uint32_t>* vec) const {
            uint64_t min_hash = 0xFFFFFFFFFFFFFFFF; // 2^64-1
            uint32_t min_token = 0;
            for (auto i : *vec) {
                auto h = hash(i);
                if (h < min_hash) {
                    min_hash = h;
                    min_token = i;
                }
            }
            return min_token;
        }
    };

    class MinHash {
    public:
        using Args = MinHashArgs;
        using Format = SetFormat;
        using Function = MinHashFunction;

    private:
        std::mt19937_64 rng;
        unsigned int set_size;

    public:
        MinHash(DatasetDimensions, unsigned int original_dimensions, Args)
            // Needs to hash to at least one bit, for which the
            // minimum set size is 2.
          : set_size(std::max(original_dimensions, 2u))
        {
            rng.seed(get_default_random_generator()());
        }

        Function sample() {
            return Function(TabulationHash(rng));
        }

        unsigned int bits_per_function() {
            return ceil_log(set_size);
        }

        float collision_probability(float similarity, int_fast8_t num_bits) {
            // Number of hashes that would collide with the given number of bits.
            float num_possible_hashes =
                static_cast<float>(set_size)/std::min(1u << num_bits, set_size)-1.0;
            // Collision probability when the lowest index is not in the intersection.
            float miss_collision_prob = num_possible_hashes/set_size;
            return similarity+(1-similarity)*miss_collision_prob;
        }
    };

    class MinHash1BitFunction {
        MinHashFunction hash;

    public: 
        MinHash1BitFunction(MinHashFunction hash)
          : hash(hash)
        {
        }

        LshDatatype operator()(std::vector<uint32_t>* vec) const {
            return hash(vec)%2;
        }
    };

    struct MinHash1Bit {
    public:
        using Args = MinHash::Args;
        using Format = MinHash::Format;
        using Function = MinHash1BitFunction;

    private:
        MinHash minhash;

    public:
        MinHash1Bit(DatasetDimensions dimensions, unsigned int original_dimensions, Args args)
          : minhash(dimensions, original_dimensions, args)
        {
        }

        Function sample() {
            return Function(minhash.sample());
        }

        unsigned int bits_per_function() {
            return 1;
        }

        float collision_probability(float similarity, int_fast8_t num_bits) {
            if (num_bits > 1) { num_bits = 1; }
            return minhash.collision_probability(similarity, num_bits);
        }
    };
}
