#pragma once

#include "dataset.hpp"
#include "ffht/fht_header_only.h"
#include "format/unit_vector.hpp"
#include "math.hpp"

namespace puffinn {
    struct CrossPolytopeCollisionEstimates {
        std::vector<std::vector<float>> probabilities;
        float eps;

        CrossPolytopeCollisionEstimates() {}

        CrossPolytopeCollisionEstimates(
            unsigned int dimensions,
            unsigned int num_repetitions,
            float eps
        )
          : eps(eps)
        {
            // adapted from https://bitbucket.org/tobc/knn_code/src/master/crosspolytopefamily.h
            std::normal_distribution<double> standard_normal(0,1);
            auto& rng = get_default_random_generator();

            auto log_dimensions = ceil_log(dimensions);
            // Number of collisions for each number of used bits
            std::vector<int> collisions(log_dimensions+2);
            probabilities = std::vector<std::vector<float>>(log_dimensions+2);

            double alpha = -1;
            //foreach [alpha, alpha+eps) segment
            while(alpha <= 1) {
                for (auto& v : collisions) { v = 0; }

                for(uint32_t i = 0; i < num_repetitions; i++) {
                    // length = dimensions
                    // x = (1, 0, ..., 0)
                    // y = (alpha, (1-alpha^2)^(1/2), ..., 0)

                    // The hash value so far.
                    uint32_t hash_x = 0;
                    uint32_t hash_y = 0;
                    // Absolute value of highest value seen.
                    double v_x = 0;
                    double v_y = 0;

                    // Compute a random rotation of x and y using the matrix z
                    // [ [ z_1_0, z_2_0 ],
                    //   [ z_1_1, z_2_1 ],
                    //   [ z_1_j, z_2_j ] ]
                    for(uint32_t j = 0; j < dimensions; j++) {
                        double z_1 = standard_normal(rng);
                        double z_2 = standard_normal(rng);
                        // calculate z*x[j] and find the index with the highest value
                        if(abs(z_1) > v_x) {
                            v_x = abs(z_1);
                            hash_x = j;
                            if (z_1 < 0) { hash_x |= (1 << log_dimensions); }
                        }
                        // do the same for z*y[j]
                        double h_y = alpha*z_1 + pow(1 - pow(alpha, 2), 0.5)*z_2;
                        if(abs(h_y) > v_y) {
                            v_y = abs(h_y);
                            hash_y = j;
                            if (h_y < 0) { hash_y |= (1 << log_dimensions); }
                        }
                    }
                    for (unsigned int used_bits = 0; used_bits <= log_dimensions+1; used_bits++) {
                        auto shift = log_dimensions+1-used_bits;
                        collisions[used_bits] += (hash_x >> shift) == (hash_y >> shift);
                    }
                }
                for (unsigned int used_bits = 0; used_bits <= log_dimensions+1; used_bits++) {
                    auto prob = static_cast<float>(collisions[used_bits])/num_repetitions;
                    probabilities[used_bits].push_back(prob);
                }
                // eps refers to the number of segments between 0 and 1, but the estimation
                // works in segments from -1 to 1.
                alpha += 2*eps;
            }
        }

        float get_collision_probability(float sim, int_fast8_t num_bits) {
            return probabilities[num_bits][(size_t)(sim/eps)];
        }
    };

    class FHTCrossPolytopeHashFunction {
        int dimensions;
        int log_dimensions;
        unsigned int num_rotations;
        // Random +-1 diagonal matrix for each rotation in each application of cross-polytope.
        // Hash idx * num_rotations * dimensions as power of 2
        // Shared_ptr as functors must be copy-constructible
        std::shared_ptr<int> random_signs;

        // Calculate a unique value depending on which axis is closest to the given floating point
        // vector.
        LshDatatype encode_closest_axis(float* vec) const {
            int res = 0;
            float max_sim = 0;
            for (int i = 0; i < (1 << log_dimensions); i++) {
                if (vec[i] > max_sim) {
                    res = i;
                    max_sim = vec[i];
                } else if (-vec[i] > max_sim) {
                    res = i+(1 << log_dimensions);
                    max_sim = -vec[i];
                }
            }
            return res;
        }

    public:
        // Create a cross polytope hasher using the given number of pseudorandom rotations
        // using hadamard transforms.
        FHTCrossPolytopeHashFunction(
            DatasetDimensions stored_dimensions,
            unsigned int num_rotations
        )
          : dimensions(stored_dimensions.actual),
            num_rotations(num_rotations)
        {
            log_dimensions = ceil_log(dimensions);

            int random_signs_len = num_rotations*(1 << log_dimensions);
            random_signs = std::unique_ptr<int>(new int[random_signs_len]);

            std::uniform_int_distribution<int_fast32_t> sign_distribution(0, 1);
            auto& generator = get_default_random_generator();
            for (int i=0; i < random_signs_len; i++) {
                random_signs.get()[i] = sign_distribution(generator)*2-1;
            }
        }

        // Hash the given vector
        LshDatatype operator()(int16_t* vec) const {
            std::unique_ptr<float> rotated_vec(new float[1 << log_dimensions]);

            // Reset rotation vec
            for (int i=0; i<dimensions; i++) {
                rotated_vec.get()[i] = UnitVectorFormat::from_16bit_fixed_point(vec[i]);
            }
            for (int i=dimensions; i < (1 << log_dimensions); i++) {
                rotated_vec.get()[i] = 0.0f;
            }

            for (unsigned int rotation = 0; rotation < num_rotations; rotation++) {
                // Multiply by a diagonal +-1 matrix.
                int sign_idx = rotation*(1 << log_dimensions);
                for (int i=0; i < (1 << log_dimensions); i++) {
                    rotated_vec.get()[i] *= random_signs.get()[sign_idx+i];
                }
                // Apply the fast hadamard transform
                fht(rotated_vec.get(), log_dimensions);
            }

            return encode_closest_axis(rotated_vec.get());
        }
    };

    // Arguments for the cross polytope lsh.
    struct FHTCrossPolytopeArgs {
        int num_rotations;
        unsigned int estimation_repetitions;
        float estimation_eps;

        constexpr FHTCrossPolytopeArgs()
            : num_rotations(3),
              estimation_repetitions(1000),
              estimation_eps(5e-3)
        {
        }
    };

    class FHTCrossPolytopeHash {
    public:
        using Args = FHTCrossPolytopeArgs;
        using Format = UnitVectorFormat;
        using Function = FHTCrossPolytopeHashFunction;

    private:
        DatasetDimensions dimensions;
        Args args;
        CrossPolytopeCollisionEstimates estimates;

    public:
        FHTCrossPolytopeHash(
            DatasetDimensions dimensions,
            unsigned int /* original_dimensions */,
            Args args
        )
          : dimensions(dimensions),
            args(args),
            estimates(
                (1 << ceil_log(dimensions.actual)),
                args.estimation_repetitions,
                args.estimation_eps)
        {
        }

        FHTCrossPolytopeHashFunction sample() {
            return FHTCrossPolytopeHashFunction(dimensions, args.num_rotations);
        }

        unsigned int bits_per_function() {
            return ceil_log(dimensions.actual)+1;
        }
 
        float collision_probability(
            float similarity,
            int_fast8_t num_bits
        ) {
            return estimates.get_collision_probability(similarity, num_bits);
        }
    };

    struct CrossPolytopeArgs {
        unsigned int estimation_repetitions;
        float estimation_eps;

        constexpr CrossPolytopeArgs()
          : estimation_repetitions(1000),
            estimation_eps(5e-3)
        {
        }
    };

    class CrossPolytopeHashFunction {
        unsigned int dimensions;
        unsigned int padded_dimensions;
        std::unique_ptr<int16_t, decltype(free)*> random_matrix;

    public:
        CrossPolytopeHashFunction(DatasetDimensions dimensions)
          : dimensions(dimensions.actual),
            padded_dimensions(dimensions.padded),
            random_matrix(
                allocate_storage<UnitVectorFormat>(
                    1 << ceil_log(dimensions.actual), 
                    dimensions.padded))
        {
            unsigned int matrix_size = (1 << ceil_log(dimensions.actual));

            for (unsigned int dim=0; dim < matrix_size; dim++) {
                auto vec = UnitVector::generate_random(dimensions.actual);
                UnitVectorFormat::store(
                    vec,
                    &random_matrix.get()[dim*padded_dimensions],
                    dimensions);
            }
        }

        LshDatatype operator()(int16_t* vec) const {
            LshDatatype res = 0;
            uint16_t max_abs_dot = 0;
            for (unsigned int i=0; i<(1u << ceil_log(dimensions)); i++) {
                auto matrix_row = &random_matrix.get()[i*padded_dimensions];
                // dot product
                auto rotated_i = dot_product_i16_avx2(vec, matrix_row, dimensions);
                if (rotated_i > max_abs_dot) {
                    max_abs_dot = rotated_i;
                    res = i;
                } else if (-rotated_i > max_abs_dot) {
                    max_abs_dot = -rotated_i;
                    res = i+(1 << ceil_log(dimensions));
                }
            }
            return res;
        }
    };

    class CrossPolytopeHash {
    public:
        using Args = CrossPolytopeArgs;
        using Format = UnitVectorFormat;
        using Function = CrossPolytopeHashFunction;

    private:
        DatasetDimensions dimensions;
        Args args;
        CrossPolytopeCollisionEstimates estimates;

    public:
        CrossPolytopeHash(
            DatasetDimensions dimensions,
            unsigned int /* original_dimensions */,
            Args args
        )
          : dimensions(dimensions),
            args(args),
            estimates(
                (1 << ceil_log(dimensions.actual)),
                args.estimation_repetitions,
                args.estimation_eps)
        {
        }

        CrossPolytopeHashFunction sample() {
            return CrossPolytopeHashFunction(dimensions);
        }

        unsigned int bits_per_function() {
            return ceil_log(dimensions.actual)+1;
        }

        float collision_probability(
            float similarity,
            int_fast8_t num_bits
        ) {
            return estimates.get_collision_probability(similarity, num_bits);
        }
    };
}
