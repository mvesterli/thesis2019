#pragma once

#include "catch.hpp"

#include "format/unit_vector.hpp"

namespace format {
    using namespace puffinn;

    TEST_CASE("UnitVector normalizing correctly") {
        UnitVector vec(std::vector<float> { 3, -4, 0});
        REQUIRE(vec[0] == Approx(3.0/5));
        REQUIRE(vec[1] == Approx(-4.0/5));
        REQUIRE(vec[2] == 0);
    }

    TEST_CASE("UnitVector::generate_random generates normalized vectors") {
        auto vec = UnitVector::generate_random(50);
        float sum = 0;
        for (float v : vec) {
            sum += v*v;
        }
        REQUIRE(sum == Approx(1));
    }

    TEST_CASE("UnitVector::generate_random even disribution") {
        const unsigned ITERATIONS = 1;
        const unsigned DIMENSIONS = 100;
        const float EPS = 0.02;

        int sums[DIMENSIONS] = { 0 };
        for (unsigned int i=0; i < ITERATIONS; i++) {
            auto vec = UnitVector::generate_random(DIMENSIONS);
            for (unsigned int j=0; j<DIMENSIONS; j++) {
                sums[j] += vec[j];
            }
        }
        for (unsigned i=0; i < DIMENSIONS; i++) {
            REQUIRE(abs(sums[i]) <= EPS);
        }
    }

    TEST_CASE("UnitVectorFormat::to_16bit_fixed_point") {
        REQUIRE(UnitVectorFormat::to_16bit_fixed_point(0.99999) == INT16_MAX);
        REQUIRE(UnitVectorFormat::to_16bit_fixed_point(1.0) == INT16_MAX);
        REQUIRE(UnitVectorFormat::to_16bit_fixed_point(-1.0) == INT16_MIN);
        REQUIRE(UnitVectorFormat::to_16bit_fixed_point(0.0) == 0);
        REQUIRE(UnitVectorFormat::to_16bit_fixed_point(0.5) == 0x4000);
        REQUIRE(UnitVectorFormat::to_16bit_fixed_point(-0.5) == (int16_t)0xc000);
    }

    TEST_CASE("UnitVectorFormat::from_16bit_fixed_point") {
        REQUIRE(UnitVectorFormat::from_16bit_fixed_point(0x0000) == 0.0);
        REQUIRE(UnitVectorFormat::from_16bit_fixed_point(0x4000) == 0.5);
        REQUIRE(UnitVectorFormat::from_16bit_fixed_point(0xa000) == -0.75);
        REQUIRE(UnitVectorFormat::from_16bit_fixed_point(0x8000) == -1.0);
        REQUIRE(UnitVectorFormat::from_16bit_fixed_point(0x7fff)
                == ((float)INT16_MAX)/(((float)INT16_MAX)+1));
    }

    TEST_CASE("pad_dimensions") {
        REQUIRE(pad_dimensions<UnitVectorFormat>(0) == 0);
        REQUIRE(pad_dimensions<UnitVectorFormat>(1) == 16);
        REQUIRE(pad_dimensions<UnitVectorFormat>(16) == 16);
        REQUIRE(pad_dimensions<UnitVectorFormat>(17) == 32);
    }
}
