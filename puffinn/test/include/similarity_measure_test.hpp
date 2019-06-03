#pragma once

#include "catch.hpp"

#include "format/generic.hpp"
#include "similarity_measure/cosine.hpp"
#include "similarity_measure/l2.hpp"
#include "similarity_measure/jaccard.hpp"

namespace similarity_measure {
    using namespace puffinn;

    TEST_CASE("CosineSimilarity::compute_similarity") {
        auto v1 = allocate_storage<UnitVectorFormat>(1, 32);
        auto v2 = allocate_storage<UnitVectorFormat>(1, 32);
        for (unsigned int i=0; i<32; i++) {
            v2.get()[i] = v1.get()[i] = 0;
        }
        v1.get()[0] = UnitVectorFormat::to_16bit_fixed_point(0.2);
        v2.get()[0] = UnitVectorFormat::to_16bit_fixed_point(0.5);
        v1.get()[21] = UnitVectorFormat::to_16bit_fixed_point(-0.85);
        v2.get()[21] = UnitVectorFormat::to_16bit_fixed_point(0.4);

        float res16 = CosineSimilarity::compute_similarity(v1.get(), v2.get(), 16);
        float res32 = CosineSimilarity::compute_similarity(v1.get(), v2.get(), 32);
        REQUIRE(std::abs(res16-0.1) <= 1e-5);
        REQUIRE(std::abs(res32-(-0.24)) <= 1e-5);
    }

    TEST_CASE("L2Distance::compute_similarity") {
        auto v1 = allocate_storage<RealVectorFormat>(1, 32);
        auto v2 = allocate_storage<RealVectorFormat>(1, 32);
        for (unsigned int i=0; i<32; i++) {
            v2.get()[i] = v1.get()[i] = 0;
        }

        v1.get()[0] = 0.2;
        v2.get()[0] = 0.5;
        v1.get()[21] = -1.5;
        v2.get()[21] = 0.4;

        float res16 = L2Distance::compute_similarity(v1.get(), v2.get(), 16);
        float res32 = L2Distance::compute_similarity(v1.get(), v2.get(), 32);
        REQUIRE(std::abs(res16-(-0.09)) <= 1e-6);
        REQUIRE(std::abs(res32-(-3.70)) <= 1e-6);
    }

    TEST_CASE("JaccardSimilarity::compute_similarity") {
        Dataset<SetFormat> dataset(100);
        auto d = dataset.get_dimensions();

        std::vector<uint32_t> a, b;
        SetFormat::store({}, &a, d);
        SetFormat::store({}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == 0);

        SetFormat::store({1, 2, 3}, &a, d);
        SetFormat::store({1, 2, 3}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == 1.0);

        SetFormat::store({1, 2, 3}, &a, d);
        SetFormat::store({4, 5, 3}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == Approx(1.0/5.0));

        SetFormat::store({1}, &a, d);
        SetFormat::store({1, 2, 3, 4, 5, 6}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == Approx(1.0/6.0));

        SetFormat::store({1, 2, 3}, &a, d);
        SetFormat::store({4, 5, 6}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == 0.0);

        SetFormat::store({5, 7, 1}, &a, d);
        SetFormat::store({1, 2, 3, 4, 5, 6}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == Approx(2.0/7.0));
    }
}


