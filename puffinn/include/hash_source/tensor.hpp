#pragma once

#include "hash_source/independent.hpp"

namespace puffinn {
    uint64_t intersperse_zero(int64_t val) {
        uint64_t mask = 1;
        uint64_t shift = 0;

        uint64_t res = 0;
        for (unsigned i=0; i < sizeof(uint64_t)*8/2; i++) {
            res |= (val & mask) << shift;
            mask <<= 1;
            shift++;
        }
        return res;
    }

    // Helper function for getting indices to tensor.
    //
    // Retrieve the pair of indices where both sides are incremented as little as possible.
    // The rhs index is incremented first
    // Eg. (0, 0) (0, 1) (1, 0) (1, 1) (0, 2)
    static std::pair<unsigned int, unsigned int> get_minimal_index_pair(int idx) {
        int sqrt = static_cast<int>(std::sqrt(idx));
        if (idx == sqrt*sqrt+2*sqrt) {
            return {sqrt, sqrt};
        } else if (idx >= sqrt*sqrt+sqrt) {
            return {sqrt, idx-(sqrt*sqrt+sqrt)};
        } else { // idx >= sqrt*sqrt, always true
            return {idx-sqrt*sqrt, sqrt};
        }
    }

    template <typename T>
    class TensoredHasher;

    // Contains two sets of hashfunctions. Hash values are constructed by interleaving one hash
    // from the first set with one from the second set. The used hash values are chosen so as
    // to avoid using the same combination twice.
    template <typename T>
    class TensoredHashSource : public HashSource<T> {
        IndependentHashSource<T> independent_hash_source;
        std::vector<std::unique_ptr<Hash>> hashers;
        std::vector<uint64_t> hashes;
        unsigned int next_hash_idx = 0;
        unsigned int num_bits;

    public:
        TensoredHashSource(
            DatasetDimensions dimensions,
            unsigned int original_dimensions,
            typename T::Args args,
            // Number of hashers to create.
            unsigned int num_hashers,
            // Number of bits per hasher.
            unsigned int num_bits
        ) 
          : independent_hash_source(
                dimensions,
                original_dimensions,
                args,
                2*std::ceil(std::sqrt(static_cast<float>(num_hashers))),
                (num_bits+1)/2),
            num_bits(num_bits)
        {
            for (unsigned int i=0; i < independent_hash_source.get_size(); i++) {
                hashers.push_back(independent_hash_source.sample());
            }
            hashes.resize(hashers.size());
        }

        std::unique_ptr<Hash> sample() {
            auto index_pair = get_minimal_index_pair(next_hash_idx);
            next_hash_idx++;
            return std::make_unique<TensoredHasher<T>>(
                this,
                index_pair.first,
                index_pair.second);
        }

        void reset(typename T::Format::Type* vec) {
            independent_hash_source.reset(vec);
            // Store the hashes so that the final hash can be created by simply bitwise or-ing them together.
            #pragma omp parallel for
            for (unsigned int i=0; i < hashers.size(); i++) {
                hashes[i] = intersperse_zero((*hashers[i])());
            }
            // Store hashes shifted by one, so that lhs hashes and rhs hashes
            // do not overlap.
            // Ensure that the lhs hashes are longer or equal to the length of the rhs hashes.
            if (num_bits%2 == 0) {
                // Shift lhs hashes
                for (unsigned int i=0; i < hashers.size()/2; i++) {
                    hashes[i] <<= 1;
                }
            } else {
                // Shift rhs hashes the other way to reduce the size as we rounded up before.
                for (unsigned int i=hashers.size()/2; i < hashers.size(); i++) {
                    hashes[i] >>= 1;
                }
            }
        }

        uint64_t hash(unsigned int lhs_idx, unsigned int rhs_idx) const {
            return hashes[lhs_idx] | hashes[hashes.size()/2+rhs_idx];
        }

        float collision_probability(
            float similarity,
            uint_fast8_t num_bits
        ) {
            return independent_hash_source.collision_probability(similarity, num_bits);
        }

        float failure_probability(
            uint_fast8_t hash_length,
            uint_fast32_t num_tables, 
            uint_fast32_t max_tables,
            float similarity
        ) {
            auto cur_left_bits = (hash_length+1)/2;
            auto cur_right_bits = hash_length-cur_left_bits;

            auto last_left_bits = (hash_length+2)/2;
            auto last_right_bits = hash_length+1-last_left_bits;

            auto cur_hashes = std::floor(std::sqrt(num_tables));
            auto last_hashes = std::floor(std::sqrt(max_tables))-cur_hashes;

            auto left_prob =
                this->concatenated_collision_probability(cur_left_bits, similarity);
            auto left_last_prob =
                this->concatenated_collision_probability(last_left_bits, similarity);

            auto right_prob =
                this->concatenated_collision_probability(cur_right_bits, similarity);
            auto right_last_prob =
                this->concatenated_collision_probability(last_right_bits, similarity);

            auto cur_upper_left_prob = 1.0-std::pow(1.0-left_prob, cur_hashes);
            auto last_upper_left_prob = 1.0-std::pow(1.0-left_last_prob, cur_hashes);
            auto last_lower_left_prob= 1.0-std::pow(1.0-left_last_prob, last_hashes);
            auto cur_upper_right_prob= 1.0-std::pow(1.0-right_prob, cur_hashes);
            auto last_upper_right_prob= 1.0-std::pow(1.0-right_last_prob, cur_hashes);
            auto last_lower_right_prob= 1.0-std::pow(1.0-right_last_prob, last_hashes);
            return
                (1-cur_upper_left_prob*cur_upper_right_prob) *
                (1-last_upper_left_prob*last_upper_right_prob) *
                (1-last_lower_left_prob*last_upper_right_prob) *
                (1-last_lower_left_prob*last_lower_right_prob);
        }

        uint_fast8_t get_bits_per_function() {
            return independent_hash_source.get_bits_per_function();
        }

        bool precomputed_hashes() const {
            return true;
        }
    };

    template <typename T>
    class TensoredHasher : public Hash {
        TensoredHashSource<T>* source;
        unsigned int lhs_idx;
        unsigned int rhs_idx;

    public:
        TensoredHasher(TensoredHashSource<T>* source, unsigned int lhs_idx, unsigned int rhs_idx)
          : source(source),
            lhs_idx(lhs_idx),
            rhs_idx(rhs_idx)
        {
        }

        uint64_t operator()() const {
            return source->hash(lhs_idx, rhs_idx);
        }
    };

    template <typename T>
    struct TensoredHashArgs : public HashSourceArgs<T> {
        typename T::Args args;

        std::unique_ptr<HashSource<T>> build(
            DatasetDimensions dimensions,
            unsigned int original_dimensions,
            unsigned int num_tables,
            unsigned int num_bits
        ) const {
            return std::make_unique<TensoredHashSource<T>> (
                dimensions,
                original_dimensions,
                args,
                num_tables,
                num_bits
            );
        }

        std::unique_ptr<HashSourceArgs<T>> copy() const {
            return std::make_unique<TensoredHashArgs<T>>(*this);
        }
    };
}
