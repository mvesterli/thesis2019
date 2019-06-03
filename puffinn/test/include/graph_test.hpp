#pragma once

#include "catch.hpp"
#include "graph.hpp"
#include "similarity_measure/cosine.hpp"

namespace graph {
    using namespace puffinn;

    TEST_CASE("Brute force - empty") {
        std::vector<std::vector<uint32_t>> expected{};
        auto res = build_graph_bf<CosineSimilarity, std::vector<float>>({}, 1);
        REQUIRE(res == expected);
    }

    TEST_CASE("Brute force - 1") {
        std::vector<std::vector<float>> points{{1,0}};
        
        std::vector<std::vector<uint32_t>> expected{{}};
        auto res = build_graph_bf<CosineSimilarity, std::vector<float>>(points, 1);
        REQUIRE(res == expected);
    }

    TEST_CASE("Brute force - multiple") {
        std::vector<std::vector<float>> points {
            {1, 0},
            {-1, 0},
            {0.5, 0.5},
            {0.4, 0.6},
            {-0.1, 1}
        };

        std::vector<std::vector<uint32_t>> expected {
            {2, 3},
            {4, 3},
            {3, 0},
            {2, 4},
            {3, 2}
        };
        auto res = build_graph_bf<CosineSimilarity, std::vector<float>>(points, 2);
        REQUIRE(res == expected);
    }

    template <typename T>
    struct MockHashFunction {
        size_t next_hash = 0;
        std::vector<uint64_t> hashes;

        MockHashFunction(std::vector<uint64_t> hashes) : hashes(hashes)
        {
        }

        uint64_t operator()(typename T::Type*) {
            auto res = hashes[next_hash];
            next_hash = (next_hash+1)%hashes.size();
            return res;
        }
    };

    struct MockHasherArgs {
        std::vector<uint64_t> hashes;
        unsigned int num_bits;
    };

    template <typename T>
    struct MockHasher {
        using Args = MockHasherArgs;
        using Format = T;
        using Function = MockHashFunction<T>; 

        std::vector<uint64_t> hashes;
        unsigned int num_bits;

        MockHasher(DatasetDimensions, unsigned int, Args args)
          : hashes(args.hashes),
            num_bits(args.num_bits)
        {
        }

        Function sample() const {
           return Function(hashes);
        }

        unsigned int bits_per_function() const {
            return num_bits;
        }

        float collision_probability(float, int_fast8_t) const {
            return 1.0; // Unused
        }
    };

    float recall(
        std::vector<std::vector<uint32_t>> exact,
        std::vector<std::vector<unsigned int>> approx
    ) {
        unsigned int total_correct = 0;
        unsigned int total = 0;
        for (size_t i=0; i < exact.size(); i++) {
            total += exact[i].size();
            for (auto neighbor : exact[i]) {
                auto count = std::count(approx[i].begin(), approx[i].end(), neighbor);
                if (count > 0) {
                    total_correct++;
                    REQUIRE(count == 1); // No duplicates
                }
            }
        }
        return static_cast<float>(total_correct)/total;
    }

/*
    TEST_CASE("Fixed buckets - augment") {
        unsigned int num_inputs = 5;

        Dataset<UnitVectorFormat> dataset(2);
        for (unsigned int i=0; i < num_inputs; i++) {
            dataset.insert(std::vector<float>{1, 0});
        }
        Result result(num_inputs, num_inputs);

        MockHasherArgs args;
        args.num_bits = 4;
        args.hashes = {
            0b0100,
            0b1111,
            0b0000,
            0b0110,
            0b0001
        };
        MockHasher<UnitVectorFormat> hash_family(dataset.get_dimensions(), 2, args);
        ConcatenatedHasher<MockHasher<UnitVectorFormat>> hasher(hash_family, 4);

        std::vector<FilterLshDatatype> sketches = {
            0b0000,
            0b0100,
            0b0110,
            0b0110,
            0b0100
        };
        std::vector<uint8_t> sketch_tresholds { 1, 1, 1, 1, 1 };

        std::vector<HashedIdx> points_buffer(num_inputs);
        Array<HashedIdx> points(points_buffer, num_inputs);

        augment_result_fixed_buckets<CosineSimilarity, MockHasher<UnitVectorFormat>>(
            result, hasher, points, dataset, 2, sketches, sketch_tresholds);
        auto augmented = result.get(); 

        REQUIRE(augmented[0] == std::vector<uint32_t>{3});
        REQUIRE(augmented[1] == std::vector<uint32_t>{});
        REQUIRE(augmented[2] == std::vector<uint32_t>{4});
        REQUIRE(augmented[3] == std::vector<uint32_t>{0});
        REQUIRE(augmented[4] == std::vector<uint32_t>{2});

        result = Result(num_inputs, num_inputs);
        std::vector<SketchedIdx> sketch_points_buffer(num_inputs);
        Array<SketchedIdx> sketch_points(sketch_points_buffer, num_inputs);
        augment_result_fixed_buckets<CosineSimilarity, MockHasher<UnitVectorFormat>>(
            result,
            hasher,
            sketch_points,
            dataset,
            2,
            sketches,
            sketch_tresholds
        );
        augmented = result.get();

        REQUIRE(augmented[0] == std::vector<uint32_t>{});
        REQUIRE(augmented[1] == std::vector<uint32_t>{});
        REQUIRE(augmented[2] == std::vector<uint32_t>{4});
        REQUIRE(augmented[3] == std::vector<uint32_t>{});
        REQUIRE(augmented[4] == std::vector<uint32_t>{2});
    }

    TEST_CASE("Fixed buckets - full") {
        unsigned int dimensions = 100;
        
        std::vector<std::vector<float>> input; 
        for (int i=0; i < 500; i++) {
            input.push_back(UnitVectorFormat::generate_random(dimensions));
        }

        auto exact_graph = build_graph_bf<CosineSimilarity>(input, 10);
        auto approx_graph = build_graph_fixed_buckets<CosineSimilarity>(input, 10, 50, 24, 10, false);
        float r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Fixed buckets recall %f\n", r);
        REQUIRE(r > 0.5);

        approx_graph = build_graph_fixed_buckets<CosineSimilarity>(input, 10, 50, 24, 10, true);
        r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Fixed buckets sketched recall %f\n", r);
        REQUIRE(r > 0.5);
    }

    TEST_CASE("Fixed hashes - augment") {
        unsigned int num_inputs = 6;

        Dataset<UnitVectorFormat> dataset(2);
        for (unsigned int i=0; i < num_inputs; i++) {
            dataset.insert(std::vector<float>{1, 0});
        }
        Result result(num_inputs, num_inputs);

        std::vector<FilterLshDatatype> sketches = {
            0b0000,
            0b0100,
            0b0110,
            0b0110,
            0b0100,
            0b0010
        };
        std::vector<uint8_t> sketch_tresholds { 1, 1, 1, 1, 1, 1 };

        MockHasherArgs args;
        args.num_bits = 2;
        args.hashes = {
            0b01,
            0b11,
            0b00,
            0b01,
            0b00,
            0b01
        };
        MockHasher<UnitVectorFormat> hash_family(dataset.get_dimensions(), 2, args);
        ConcatenatedHasher<MockHasher<UnitVectorFormat>> hasher(hash_family, 2);

        std::vector<HashedIdx> points_buffer(num_inputs+1);
        Array<HashedIdx> points(points_buffer, num_inputs);

        augment_result_fixed_hash<CosineSimilarity, MockHasher<UnitVectorFormat>>(
            result, hasher, points, dataset, sketches, sketch_tresholds);
        auto augmented = result.get(); 

        REQUIRE(augmented[0] == std::vector<uint32_t>{5, 3});
        REQUIRE(augmented[1] == std::vector<uint32_t>{});
        REQUIRE(augmented[2] == std::vector<uint32_t>{4});
        REQUIRE(augmented[3] == std::vector<uint32_t>{5, 0});
        REQUIRE(augmented[4] == std::vector<uint32_t>{2});
        REQUIRE(augmented[5] == std::vector<uint32_t>{3, 0});

        result = Result(num_inputs, num_inputs);
        std::vector<SketchedIdx> sketch_points_buffer(num_inputs+1);
        Array<SketchedIdx> sketch_points(sketch_points_buffer, num_inputs);

        augment_result_fixed_hash<
            CosineSimilarity,
            MockHasher<UnitVectorFormat>
        >(
            result,
            hasher,
            sketch_points,
            dataset,
            sketches,
            sketch_tresholds
        );
        augmented = result.get();

        REQUIRE(augmented[0] == std::vector<uint32_t>{5});
        REQUIRE(augmented[1] == std::vector<uint32_t>{});
        REQUIRE(augmented[2] == std::vector<uint32_t>{4});
        REQUIRE(augmented[3] == std::vector<uint32_t>{5});
        REQUIRE(augmented[4] == std::vector<uint32_t>{2});
        REQUIRE(augmented[5] == std::vector<uint32_t>{3, 0});
    }

    TEST_CASE("Fixed hash - full") {
        unsigned int dimensions = 100;
        
        std::vector<std::vector<float>> input; 
        for (int i=0; i < 500; i++) {
            input.push_back(UnitVectorFormat::generate_random(dimensions));
        }

        auto exact_graph = build_graph_bf<CosineSimilarity>(input, 10);
        auto approx_graph = build_graph_fixed_hash<CosineSimilarity>(input, 10, 0.5, 9, false);
        float r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Fixed hash recall %f\n", r);
        REQUIRE(r > 0.5);

        approx_graph = build_graph_fixed_hash<CosineSimilarity>(input, 10, 0.5, 9, true);
        r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Fixed hash sketched recall %f\n", r);
        REQUIRE(r > 0.5);
    }

    TEST_CASE("Varying hash - augment") {
        unsigned int num_inputs = 5;

        Dataset<UnitVectorFormat> dataset(2);
        for (unsigned int i=0; i < num_inputs; i++) {
            dataset.insert(std::vector<float>{1, 0});
        }
        Result result(num_inputs, num_inputs);

        std::vector<FilterLshDatatype> sketches = {
            0b0000,
            0b0111,
            0b0011,
            0b0001,
            0b0010
        };
        std::vector<uint8_t> sketch_tresholds { 1, 1, 1, 1, 1 };

        MockHasherArgs args;
        args.num_bits = 2;
        args.hashes = {
            0b01,
            0b11,
            0b00,
            0b01,
            0b00
        };
        MockHasher<UnitVectorFormat> hash_family(dataset.get_dimensions(), 2, args);
        ConcatenatedHasher<MockHasher<UnitVectorFormat>> hasher(hash_family, 2);

        std::vector<uint8_t> hash_lengths = { 2, 2, 1, 1, 2 };
        std::vector<VaryingHashIdx> points_buffer(num_inputs+1);
        Array<VaryingHashIdx> points(points_buffer, num_inputs);
        
        augment_result_varying_hash<CosineSimilarity, MockHasher<UnitVectorFormat>>(
            result, hasher, points, dataset, hash_lengths, 
            sketches, sketch_tresholds);
        auto augmented = result.get(); 

        REQUIRE(augmented[0] == std::vector<uint32_t>{3, 2});
        REQUIRE(augmented[1] == std::vector<uint32_t>{});
        REQUIRE(augmented[2] == std::vector<uint32_t>{4, 3, 0});
        REQUIRE(augmented[3] == std::vector<uint32_t>{2, 0});
        REQUIRE(augmented[4] == std::vector<uint32_t>{2});

        result = Result(num_inputs, num_inputs);
        std::vector<SketchedVaryingHashIdx> sketch_points_buffer(num_inputs+1);
        Array<SketchedVaryingHashIdx> sketch_points(sketch_points_buffer, num_inputs);

        augment_result_varying_hash<
            CosineSimilarity,
            MockHasher<UnitVectorFormat>
        >(
            result,
            hasher,
            sketch_points,
            dataset,
            hash_lengths,
            sketches,
            sketch_tresholds
        );
        augmented = result.get();
        REQUIRE(augmented[0] == std::vector<uint32_t>{3});
        REQUIRE(augmented[1] == std::vector<uint32_t>{});
        REQUIRE(augmented[2] == std::vector<uint32_t>{4, 3});
        REQUIRE(augmented[3] == std::vector<uint32_t>{2, 0});
        REQUIRE(augmented[4] == std::vector<uint32_t>{2});
    }

    TEST_CASE("Equal failure prob - full") {
        unsigned int dimensions = 100;
        
        std::vector<std::vector<float>> input; 
        for (int i=0; i < 500; i++) {
            input.push_back(UnitVectorFormat::generate_random(dimensions));
        }

        auto exact_graph = build_graph_bf<CosineSimilarity>(input, 10);
        auto approx_graph = build_graph_equal_failure_prob<CosineSimilarity, std::vector<float>, SimHash, SimHash>(
            input, 10, 0.5, 24, 0.1, false);
        float r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Equal failure prob recall %f\n", r);
        REQUIRE(r > 0.5);

        approx_graph = build_graph_equal_failure_prob<CosineSimilarity>(
            input, 10, 0.5, 24, 0.1, true);
        r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Equal failure prob sketched recall %f\n", r);
        REQUIRE(r > 0.5);
    }

    TEST_CASE("projected hash - augment") {
        unsigned int num_inputs = 5;

        Dataset<UnitVectorFormat> dataset(2);
        for (unsigned int i=0; i < num_inputs; i++) {
            dataset.insert(std::vector<float>{1, 0});
        }
        Result result(num_inputs, num_inputs);

        std::vector<FilterLshDatatype> sketches = {
            0b0000,
            0b0111,
            0b0011,
            0b0001,
            0b0010
        };
        std::vector<uint8_t> sketch_tresholds { 1, 1, 1, 1, 1 };

        MockHasherArgs args;
        args.num_bits = 3;
        args.hashes = {
            0b010,
            0b110,
            0b001,
            0b010,
            0b000
        };
        MockHasher<UnitVectorFormat> hasher(dataset.get_dimensions(), 2, args);

        // weights are given with the least significant digit first.
        std::vector<float> projection{4, -0.5, 1};

        std::vector<ProjectionIdx> points_buffer(num_inputs);
        Array<ProjectionIdx> points(points_buffer, num_inputs);
        augment_result_projection<CosineSimilarity, MockHasher<UnitVectorFormat>>(
            result, hasher, points, dataset, 3, projection, 2,
            sketches, sketch_tresholds);
        auto augmented = result.get(); 

        REQUIRE(augmented[0] == std::vector<uint32_t>{3});
        REQUIRE(augmented[1] == std::vector<uint32_t>{4});
        REQUIRE(augmented[2] == std::vector<uint32_t>{});
        REQUIRE(augmented[3] == std::vector<uint32_t>{0});
        REQUIRE(augmented[4] == std::vector<uint32_t>{1});

        result = Result(num_inputs, num_inputs);
        std::vector<SketchedProjectionIdx> sketch_points_buffer(num_inputs);
        Array<SketchedProjectionIdx> sketch_points(sketch_points_buffer, num_inputs);

        augment_result_projection<
            CosineSimilarity,
            MockHasher<UnitVectorFormat>
        >(
            result,
            hasher,
            sketch_points,
            dataset,
            3,
            projection,
            2,
            sketches,
            sketch_tresholds
        );
        augmented = result.get();
        REQUIRE(augmented[0] == std::vector<uint32_t>{3});
        REQUIRE(augmented[1] == std::vector<uint32_t>{});
        REQUIRE(augmented[2] == std::vector<uint32_t>{});
        REQUIRE(augmented[3] == std::vector<uint32_t>{0});
        REQUIRE(augmented[4] == std::vector<uint32_t>{});
    }

    TEST_CASE("projected hash full") {
        unsigned int dimensions = 100;
        
        std::vector<std::vector<float>> input; 
        for (int i=0; i < 500; i++) {
            input.push_back(UnitVectorFormat::generate_random(dimensions));
        }

        auto exact_graph = build_graph_bf<CosineSimilarity>(input, 10);
        auto approx_graph = build_graph_projection<CosineSimilarity>(
                input, 10, 50, 24, 16, false);
        float r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Projection recall %f\n", r);
        REQUIRE(r > 0.5);

        approx_graph = build_graph_projection<CosineSimilarity>(
            input, 10, 50, 24, 16, true);
        r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Projection sketched recall %f\n", r);
        REQUIRE(r > 0.5);
    }

    TEST_CASE("NN-descent full") {
        unsigned int dimensions = 100;
        
        std::vector<std::vector<float>> input; 
        for (int i=0; i < 500; i++) {
            input.push_back(UnitVectorFormat::generate_random(dimensions));
        }

        auto exact_graph = build_graph_bf<CosineSimilarity>(input, 10);
        auto approx_graph = build_graph_nndescent<CosineSimilarity>(
                input, 10, 50, 1.0, false);
        float r = recall(exact_graph, approx_graph);
        fprintf(stderr, "nn-descent recall %f\n", r);
        REQUIRE(r > 0.5);

        approx_graph = build_graph_nndescent<CosineSimilarity>(
                input, 10, 50, 1.0, true);
        r = recall(exact_graph, approx_graph);
        fprintf(stderr, "nn-descent sketched recall %f\n", r);
        REQUIRE(r > 0.5);
    }

    TEST_CASE("NN-descent - augment") {
        unsigned int num_inputs = 5;

        Dataset<UnitVectorFormat> dataset(2);
        for (unsigned int i=0; i < num_inputs; i++) {
            dataset.insert(std::vector<float>{1, 0});
        }
        Result result(num_inputs, num_inputs);
        auto sim = CosineSimilarity::compute_similarity(dataset[0], dataset[0], 2);

        result.insert(0, 1, sim);
        result.insert(0, 2, sim);
        result.insert(1, 3, sim);
        result.insert(3, 4, sim);

        JoinSets join_sets(num_inputs, num_inputs, 1);
        augment_result_nndescent<CosineSimilarity>(result, dataset, join_sets);

        auto augmented = result.get(); 
        REQUIRE(augmented[0] == std::vector<uint32_t>{3, 2, 1});
        REQUIRE(augmented[1] == std::vector<uint32_t>{4, 3, 2, 0});
        REQUIRE(augmented[2] == std::vector<uint32_t>{1, 0});
        REQUIRE(augmented[3] == std::vector<uint32_t>{4, 1, 0});
        REQUIRE(augmented[4] == std::vector<uint32_t>{3, 1});

        std::vector<FilterLshDatatype> sketches = {
            0b0000,
            0b0111,
            0b0011,
            0b0001,
            0b0010
        };
        std::vector<uint8_t> sketch_tresholds { 1, 1, 1, 1, 1 };
        result = Result(num_inputs, num_inputs);

        result.insert(0, 1, sim);
        result.insert(0, 2, sim);
        result.insert(1, 3, sim);
        result.insert(3, 4, sim);

        augment_result_nndescent_with_sketching<CosineSimilarity>(
            result, dataset, join_sets, sketches, sketch_tresholds);

        augmented = result.get();
        REQUIRE(augmented[0] == std::vector<uint32_t>{3, 2, 1});
        REQUIRE(augmented[1] == std::vector<uint32_t>{3, 2, 0});
        REQUIRE(augmented[2] == std::vector<uint32_t>{1, 0});
        REQUIRE(augmented[3] == std::vector<uint32_t>{4, 1, 0});
        REQUIRE(augmented[4] == std::vector<uint32_t>{3});
    }
*/

    TEST_CASE("Varying repetitions - full") {
        unsigned int dimensions = 100;
        
        std::vector<std::vector<float>> input; 
        for (int i=0; i < 500; i++) {
            input.push_back(UnitVectorFormat::generate_random(dimensions));
        }

        auto exact_graph = build_graph_bf<CosineSimilarity>(input, 10);
        auto approx_graph = build_graph_varying_repetitions<CosineSimilarity, std::vector<float>>(
            input, 10, 24, 0.5, 0.01, false, false);
        float r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Varying repetitions recall %f\n", r);
        REQUIRE(r > 0.5);

        approx_graph = build_graph_varying_repetitions<CosineSimilarity>(
                input, 10, 24, 0.5, 0.01, true, false);
        r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Varying repetitions recall sketched recall %f\n", r);
        REQUIRE(r > 0.5);

        approx_graph = build_graph_varying_repetitions<CosineSimilarity>(
                input, 10, 24, 0.5, 0.01, false, true);
        r = recall(exact_graph, approx_graph);
        fprintf(stderr, "Varying repetitions recall frozen recall %f\n", r);
        REQUIRE(r > 0.5);
    }
}
