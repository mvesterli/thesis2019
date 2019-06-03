#pragma once

#include "catch.hpp"
#include "collection.hpp"
#include "hash/simhash.hpp"
#include "hash/crosspolytope.hpp"
#include "hash_source/pool.hpp"
#include "hash_source/independent.hpp"
#include "hash_source/tensor.hpp"
#include "similarity_measure/cosine.hpp"
#include "similarity_measure/jaccard.hpp"

namespace collection {
    using namespace puffinn;

    const unsigned int MB = 1024*1024;

    TEST_CASE("LSHTable::search_bf_k") {
        const unsigned DIMENSIONS = 2;

        std::vector<UnitVector> inserted {
            UnitVector({1, 0}),
            UnitVector({-1, -1}),
            UnitVector({1, 0.15}),
            UnitVector({1, 0.2}),
            UnitVector({1, -0.1}),
        };

        LSHTable<CosineSimilarity> table(DIMENSIONS, 1*MB);
        for (auto &vec : inserted) {
            table.insert(vec);
        }
        // No rebuilding necessary

        UnitVector query({1, 0});

        SECTION("k = 0") {
            REQUIRE(table.search_bf_k(query, 0).size() == 0);
        }

        SECTION("k = 1") {
            auto res = table.search_bf_k(query, 1);
            REQUIRE(res.size() == 1);
            REQUIRE(res[0] == 0);
        }

        SECTION("k = 2") {
            auto res = table.search_bf_k(query, 2);
            REQUIRE(res.size() == 2);
            REQUIRE(res[0] == 0);
            REQUIRE(res[1] == 4);
        }

        SECTION("k = 5") {
            auto res = table.search_bf_k(query, 5);
            REQUIRE(res.size() == 5);
            REQUIRE(res[0] == 0);
            REQUIRE(res[1] == 4);
            REQUIRE(res[2] == 2);
            REQUIRE(res[3] == 3);
            REQUIRE(res[4] == 1);
        }

        SECTION("k > size") {
            REQUIRE(table.search_bf_k(query, 10).size() == 5);
        }
    }

    template <typename T, typename U>
    void test_angular_search_k(
        int n,
        int dimensions,
        std::unique_ptr<HashSourceArgs<T>> hash_source = std::unique_ptr<HashSourceArgs<T>>()
    ) {
        const int NUM_SAMPLES = 100;

        std::vector<float> recalls = {0.2, 0.5, 0.95};
        std::vector<unsigned int> ks = {1, 10};

        std::vector<UnitVector> inserted;
        for (int i=0; i<n; i++) {
            inserted.push_back(UnitVector::generate_random(dimensions));
        }

        LSHTable<CosineSimilarity, T, U> table(dimensions, 100*MB);
        if (hash_source) {
            table = LSHTable<CosineSimilarity, T, U>(dimensions, 100*MB, *hash_source);
        }
        for (auto &vec : inserted) {
            table.insert(vec);
        }
        table.rebuild();

        for (auto k : ks) {
            for (auto recall : recalls) {
                int num_correct = 0;
                auto adjusted_k = std::min(k, table.get_size());
                float expected_correct = recall*adjusted_k*NUM_SAMPLES;
                for (int sample=0; sample < NUM_SAMPLES; sample++) {
                    auto query = UnitVector::generate_random(table.get_dimensions());
                    auto exact = table.search_bf_k(query, k);
                    auto res = table.search_k(query, k, recall);

                    REQUIRE(res.size() == static_cast<size_t>(adjusted_k));
                    for (auto i : exact) {
                        // Each expected value is returned once.
                        if (std::count(res.begin(), res.end(), i) != 0) {
                            num_correct++;
                        }
                    }
                }
                REQUIRE(num_correct >= expected_correct);
            }
        }
    }

    TEST_CASE("LSHTable::search_k - empty") {
        test_angular_search_k<SimHash, SimHash>(0, 2);
    }

    TEST_CASE("LSHTable::search_k - 1 value") {
        test_angular_search_k<SimHash, SimHash>(1, 5);
    }

    TEST_CASE("LSHTable::search_k simhash") {
        std::vector<int> dimensions = {5, 100};

        for (auto d : dimensions) {
            std::unique_ptr<HashSourceArgs<SimHash>> args =
                std::make_unique<HashPoolArgs<SimHash>>(3000);
            test_angular_search_k<SimHash, SimHash>(500, d, std::move(args));

            args = std::make_unique<IndependentHashArgs<SimHash>>();
            test_angular_search_k<SimHash, SimHash>(500, d, std::move(args)); 

            args = 
 std::make_unique<TensoredHashArgs<SimHash>>();            
            test_angular_search_k<SimHash, SimHash>(500, d, std::move(args));
        }
    }

    TEST_CASE("LSHTable::search_k fht cross-polytope") {
        std::vector<int> dimensions = {5, 100};

        for (auto d : dimensions) {
            std::unique_ptr<HashSourceArgs<FHTCrossPolytopeHash>> args =
                std::make_unique<HashPoolArgs<FHTCrossPolytopeHash>>(3000);
            test_angular_search_k<FHTCrossPolytopeHash, SimHash>(500, d, std::move(args));

            args = std::make_unique<IndependentHashArgs<FHTCrossPolytopeHash>>();
            test_angular_search_k<FHTCrossPolytopeHash, SimHash>(500, d, std::move(args));


            args = std::make_unique<TensoredHashArgs<FHTCrossPolytopeHash>>();
            test_angular_search_k<FHTCrossPolytopeHash, SimHash>(500, d, std::move(args));
        }
    }

    void test_jaccard_search_k(
        int n,
        int dimensions,
        std::unique_ptr<HashSourceArgs<MinHash>> hash_source =
            std::unique_ptr<HashSourceArgs<MinHash>>()
    ) {
        const int NUM_SAMPLES = 100;

        std::vector<float> recalls = {0.2, 0.5, 0.95};
        std::vector<unsigned int> ks = {1, 10};

        std::vector<std::vector<uint32_t>> inserted;
        for (int i=0; i<n; i++) {
            inserted.push_back(SetFormat::generate_random(dimensions));
        }

        LSHTable<JaccardSimilarity> table(dimensions, 100*MB);
        if (hash_source) {
            table = LSHTable<JaccardSimilarity>(dimensions, 100*MB, *hash_source);
        }
        for (auto &vec : inserted) {
            table.insert(vec);
        }
        table.rebuild();

        for (auto k : ks) {
            for (auto recall : recalls) {
                int num_correct = 0;
                auto adjusted_k = std::min(k, table.get_size());
                float expected_correct = recall*adjusted_k*NUM_SAMPLES;
                for (int sample=0; sample < NUM_SAMPLES; sample++) {
                    auto query = SetFormat::generate_random(dimensions);
                    auto exact = table.search_bf_k(query, k);
                    auto res = table.search_k(query, k, recall, FilterType::None);

                    REQUIRE(res.size() == static_cast<size_t>(adjusted_k));
                    for (auto i : exact) {
                        // Each expected value is returned once.
                        if (std::count(res.begin(), res.end(), i) != 0) {
                            num_correct++;
                        }
                    }
                }
                REQUIRE(num_correct >= expected_correct);
            }
        }
    }

    TEST_CASE("LSHTable::search_k minhash") {
        std::vector<int> dimensions = {100};

        for (auto d : dimensions) {
            std::unique_ptr<HashSourceArgs<MinHash>> args =
                std::make_unique<HashPoolArgs<MinHash>>(3000);
            test_jaccard_search_k(500, d, std::move(args));

            args = std::make_unique<IndependentHashArgs<MinHash>>();
            test_jaccard_search_k(500, d, std::move(args));


            args = std::make_unique<TensoredHashArgs<MinHash>>();
            test_jaccard_search_k(500, d, std::move(args));
        }
    }
}
