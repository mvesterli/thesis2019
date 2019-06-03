#pragma once

#include "catch.hpp"
#include "similarity_measure/cosine.hpp"
#include "maxbuffer.hpp"

namespace maxbuffer {
    using namespace puffinn;

    TEST_CASE("Constructed with size 0") {
        MaxBuffer buffer(0);
        buffer.insert(1, 1);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{});
    }

    TEST_CASE("Empty MaxBuffer") {
        MaxBuffer buffer(2);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{});
    }

    TEST_CASE("Single element") {
        MaxBuffer buffer(2);
        buffer.insert(2, 0);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{2, 0}});
        REQUIRE(buffer.smallest_value() == 0);
    }

    TEST_CASE("Retrieve while empty") {
        MaxBuffer buffer(2);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{});
        buffer.insert(100, -1000);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{100, -1000}});
        REQUIRE(buffer.smallest_value() == -1000);
    }

    TEST_CASE("Retrieve before filter") {
        MaxBuffer buffer(2);
        buffer.insert(100, 1);
        buffer.insert(50, 2);
        buffer.insert(105, 3);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{105, 3}, {50, 2}});
        REQUIRE(buffer.smallest_value() == 2);
    }

    TEST_CASE("Multiple filters") {
        MaxBuffer buffer(2);
        buffer.insert(1, 1);
        buffer.insert(2, 5);
        buffer.insert(3, -3);
        buffer.insert(4, 0);
        buffer.insert(5, 5);
        buffer.insert(6, 9);
        buffer.insert(7, 7);
        buffer.insert(8, 8);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{6, 9}, {8, 8}});
        REQUIRE(buffer.smallest_value() == 8);
    }

    TEST_CASE("Deduplication") {
        MaxBuffer buffer(2);
        buffer.insert(1, 1);
        buffer.insert(1, 1);
        buffer.insert(1, 1);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{1, 1}});
    }
}
