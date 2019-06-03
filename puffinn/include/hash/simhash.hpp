#pragma once

#include "dataset.hpp"
#include "format/unit_vector.hpp"
#include "math.hpp"

namespace puffinn {
    class SimHashFunction {
        std::unique_ptr<typename UnitVectorFormat::Type, decltype(free)*> hash_vec;
        unsigned int dimensions;

    public:
        SimHashFunction(DatasetDimensions dimensions)
          : hash_vec(allocate_storage<UnitVectorFormat>(1, dimensions.padded)),
            dimensions(dimensions.actual)
        {
            auto vec = UnitVector::generate_random(dimensions.actual);
            UnitVectorFormat::store(vec, hash_vec.get(), dimensions);
        }

        // Hash the given vector.
        LshDatatype operator()(int16_t* vec) const {
            auto dot = dot_product_i16_avx2(hash_vec.get(), vec, dimensions);
            return dot >= UnitVectorFormat::to_16bit_fixed_point(0.0);
        }
    };

    // SimHash does not take any arguments.
    struct SimHashArgs {};

    class SimHash {
    public:
        using Args = SimHashArgs;
        using Format = UnitVectorFormat;
        using Function = SimHashFunction;

    private:
        DatasetDimensions dimensions;

    public:
        SimHash(DatasetDimensions dimensions, unsigned int /* original_dimensions*/, Args)
          : dimensions(dimensions)
        {
        }

        SimHashFunction sample() {
            return SimHashFunction(dimensions);
        }

        unsigned int bits_per_function() {
            return 1;
        }

        float collision_probability(float similarity, int_fast8_t) {
            return 1.0-std::acos(2*similarity-1)/M_PI;
        }
    };
}
