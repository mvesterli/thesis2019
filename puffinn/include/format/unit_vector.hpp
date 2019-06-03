#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#include "typedefs.hpp"

namespace puffinn {
    class UnitVector {
        std::vector<float> values;
    public:
        UnitVector(std::vector<float> values) {
            float len_squared = 0;
            for (auto num : values) { len_squared += num*num; }
            float len = std::sqrt(len_squared);
            if (len != 0.0) {
                for (auto &num : values) { num /= len; }
            }
            this->values = values;
        }

        // Generate a random vector.
        static UnitVector generate_random(int dimensions) {
            std::normal_distribution<float> normal_distribution(0.0, 1.0);
            auto& generator = get_default_random_generator();
            std::vector<float> values;
            for (int i=0; i<dimensions; i++) {
                values.push_back(normal_distribution(generator));
            }
            return UnitVector(values);
        }

        // A simple dot product between the two vectors.
        float dot(const UnitVector& rhs) const {
            assert(get_dimensions() == rhs.get_dimensions());

            float res = 0.0;
            for (unsigned i=0; i<get_dimensions(); i++) {
                res += values[i]*rhs[i];
            }
            return res;
        }

        // Retrieve the number of dimensions in the vector.
        unsigned int get_dimensions() const {
            return values.size();
        }

        std::vector<float>::iterator begin() {
            return values.begin();
        }

        std::vector<float>::iterator end() {
            return values.end();
        }

        // Access the n'th element of the vector.
        const float& operator[](unsigned int idx) const {
            return values[idx];
        }
    };

    // Bytes in a 256-bit vector
    const int VECTOR256_ALIGNMENT = 256/8;

    // Store vectors in 256-bit aligned vectors of fixed-point values between -1 and 1.
    struct UnitVectorFormat {
        // Represent the values as signed 15bit fixed point numbers between -1 and 1.
        // Done since the values are always in the range [-1, 1].
        // This is equivalent to what is used by `mulhrs`. However this cannot represent 1 exactly.
        using Type = int16_t;

        const static unsigned int ALIGNMENT = VECTOR256_ALIGNMENT;

        // Number of `Type` values that fit into a 256 bit vector.
        const static unsigned int VALUES_PER_VEC = 16;

        // Convert a floating point value between -1 and 1 to the internal, fixed point representation.
        static constexpr int16_t to_16bit_fixed_point(float val) {
            assert(val >= -1.0 && val <= 1.0);
            // The value cannot represent 1, so for accuracy, positive numbers are compared to the value
            // above INT16_MAX. 1.0 is set to the greatest representable value.
            auto res = (int16_t)(-val*INT16_MIN);
            if (val > 0.0 && res < 0) { res = INT16_MAX; }
            return res;
        }

        // Convert a number between -1 and 1 from the internal,
        // fixed point representation to floating point.
        static constexpr float from_16bit_fixed_point(Type val) {
            return -((float)val)/INT16_MIN;
        }

        static unsigned int storage_dimensions(unsigned int dimensions) {
            return dimensions;
        }

        static void store(
            const UnitVector& input,
            Type* storage,
            DatasetDimensions dimensions
        ) {
            if (input.get_dimensions() != dimensions.actual) {
                throw "Invalid size";
            }
            for (unsigned int i=0; i<input.get_dimensions(); i++) {
                storage[i] = to_16bit_fixed_point(input[i]);
            }
            for (unsigned int i=input.get_dimensions(); i < dimensions.padded; i++) {
                storage[i] = to_16bit_fixed_point(0.0);
            }
        }

        static void store(
            const std::vector<float>& input,
            Type* storage,
            DatasetDimensions dimensions
        ) {
            store(UnitVector(input), storage, dimensions);
        }

        static std::vector<float> generate_random(unsigned int dimensions) {
            std::normal_distribution<float> normal_distribution(0.0, 1.0);
            auto& generator = get_default_random_generator();

            std::vector<float> values;
            float len_squared = 0;
            for (int i=0; i<dimensions; i++) {
                auto val = normal_distribution(generator);
                values.push_back(val);
                len_squared += val*val;
            }
            float len = std::sqrt(len_squared);
            for (auto &num : values) { num /= len; }
            return values;
        }
    };
}
