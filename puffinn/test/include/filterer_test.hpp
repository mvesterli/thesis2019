#pragma once

#include "catch.hpp"
#include "filterer.hpp"
#include "format/unit_vector.hpp"
#include "hash_source/independent.hpp"
#include "hash/simhash.hpp"

using namespace puffinn;

namespace filterer_test {
    TEST_CASE("filtering equal/opposite vector") {
        Dataset<UnitVectorFormat> dataset(2);
        dataset.insert(UnitVector({1, 0}));
        dataset.insert(UnitVector({-1, 0}));

        IndependentHashArgs<SimHash> hash_args;
        auto hash_source = hash_args.build(
            dataset.get_dimensions(),
            2,
            NUM_SKETCHES,
            NUM_FILTER_HASHBITS);
        Filterer<SimHash> filterer(std::move(hash_source));

        filterer.add_sketches(dataset);

        UnitVector query({1, 0});
        auto stored = to_stored_type<UnitVectorFormat>(query, dataset.get_dimensions());
        filterer.reset(stored.get());

        // Anything initially passes
        for (size_t i=0; i < NUM_SKETCHES; i++) {
            REQUIRE(filterer.passes_filter(1, i));
        }

        filterer.update_max_sketch_diff(1.0);
        for (size_t i=0; i < NUM_SKETCHES; i++) {
            REQUIRE(filterer.passes_filter(0, i));
            REQUIRE(!filterer.passes_filter(1, i));
        }

        filterer.update_max_sketch_diff(-1.0);
        for (size_t i=0; i < NUM_SKETCHES; i++) {
            REQUIRE(filterer.passes_filter(1, i));
        }
    }

    TEST_CASE("All filtering bits used") {
        const int NUM_VECTORS = 20;
        const unsigned int DIMENSIONS = 100;

        Dataset<UnitVectorFormat> dataset(DIMENSIONS);
        for (int i=0; i < NUM_VECTORS; i++) {
            dataset.insert(UnitVector::generate_random(DIMENSIONS));
        }

        IndependentHashArgs<SimHash> hash_args;
        auto hash_source = hash_args.build(
            dataset.get_dimensions(),
            DIMENSIONS,
            NUM_SKETCHES,
            NUM_FILTER_HASHBITS);
        Filterer<SimHash> filterer(std::move(hash_source));
        filterer.add_sketches(dataset);

        int bit_counts[NUM_FILTER_HASHBITS];
        for (int idx=0; idx < NUM_VECTORS; idx++) {
            for (unsigned int sketch=0; sketch < NUM_SKETCHES; sketch++) {
                for (unsigned int bit=0; bit < NUM_FILTER_HASHBITS; bit++) {
                    if (filterer.get_sketch(idx, sketch) & (1 << bit)) {
                        bit_counts[bit]++;
                    }
                }
            }
        }
        for (unsigned int bit=0; bit < NUM_FILTER_HASHBITS; bit++) {
            REQUIRE(bit_counts[bit] != 0);
        }
    }
}
