#pragma once

#include "graph/common.hpp"

namespace puffinn {
    LshDatatype gray_to_binary(LshDatatype val) {
        LshDatatype mask = val >> 1;
        while (mask != 0) {
            val ^= mask;
            mask >>= 1;
        }
        return val;
    }
    
    template <typename TSim, typename THash, typename T, typename U>
    void augment_result_fixed_buckets(
        Result& result,
        ConcatenatedHasher<THash>& hasher,
        Array<T> points,
        const Dataset<typename TSim::Format>& dataset,
        unsigned int bucket_size,
        const std::vector<U>& sketches,
        const std::vector<uint16_t>& sketch_tresholds,
        bool use_graycodes = false
    ) {
        HashedIdx::prepare_points(points);
        T::set_sketches(points, sketches, sketch_tresholds);
        compute_fixed_hashes(points, hasher, dataset);
        if (use_graycodes) {
            for (auto& point : points) {
                point.hash = gray_to_binary(point.hash);
            }
        }
        std::sort(points.begin(), points.end(), [](T& a, T& b) {
            return a.hash < b.hash;
        });

        g_performance_metrics.start_timer(Computation::Augment);
        for (
            size_t start_idx = 0;
            start_idx < points.len-(bucket_size-1);
            start_idx += bucket_size
        ) {
            g_performance_metrics.add_candidates(bucket_size*(bucket_size-1)/2);
            for (size_t i=start_idx; i < start_idx+bucket_size; i++) {
                auto real_i = points[i].idx;
                for (size_t j=i+1; j < start_idx+bucket_size; j++) {
                    auto real_j = points[j].idx;
                    if (points[i].passes_sketch(points[j])) {
                        g_performance_metrics.add_distance_computations(1);
                        auto sim = TSim::compute_similarity(
                            dataset[real_i],
                            dataset[real_j],
                            dataset.get_dimensions().actual);
                        result.insert(real_i, real_j, sim);
                    }
                }
            }
        }
        g_performance_metrics.store_time(Computation::Augment);
    }
    
    template <
        typename TSim,
        typename U,
        typename TSketchLen = Sketch64Bit,
        typename THash = typename TSim::DefaultHash,
        typename TSketch = typename TSim::DefaultSketch
    >
    std::vector<std::vector<uint32_t>> build_graph_fixed_buckets(
        const std::vector<U>& vectors,
        unsigned int k,
        unsigned int num_repetitions,
        unsigned int num_bits,
        unsigned int bucket_size,
        bool use_sketching,
        bool use_graycodes = false,
        typename THash::Args hash_args = typename THash::Args(),
        typename TSketch::Args sketch_args = typename TSketch::Args()
    ) {
        if (vectors.size() == 0) {
            return {};
        }
        g_performance_metrics.new_query();
        g_performance_metrics.start_timer(Computation::Total);
        g_performance_metrics.start_timer(Computation::Setup);
        auto dimensions = vectors[0].size();

        Dataset<typename TSim::Format> dataset(dimensions, vectors.size());
        for (const auto& v : vectors) {
            dataset.insert(v);
        }

        Result result(vectors.size(), k);
        THash hash_family(dataset.get_dimensions(), dimensions, hash_args);
        g_performance_metrics.store_time(Computation::Setup);

        if (use_sketching) {
            const unsigned int NUM_SKETCHES = 16;
            TSketch sketch_family(dataset.get_dimensions(), dimensions, sketch_args);
            
            std::vector<SketchedIdx<TSketchLen>> points_buffer(vectors.size());
            Array<SketchedIdx<TSketchLen>> points(points_buffer, vectors.size());
            auto sketches = generate_sketches<TSketchLen>(sketch_family, dataset, NUM_SKETCHES);
            std::vector<uint16_t> sketch_tresholds(vectors.size(), NUM_FILTER_HASHBITS);

            for (unsigned int i=0; i < num_repetitions; i++) {
                ConcatenatedHasher<THash> hasher(hash_family, num_bits);
                augment_result_fixed_buckets<TSim>(
                    result, hasher, points, dataset, bucket_size,
                    sketches[i%NUM_SKETCHES], sketch_tresholds, use_graycodes);
                update_sketch_tresholds(
                    sketch_family, result, sketch_tresholds, TSketchLen::length());
            }
        } else {
            std::vector<HashedIdx> points_buffer(vectors.size());
            Array<HashedIdx> points(points_buffer, vectors.size());
            std::vector<FilterLshDatatype> sketches;
            std::vector<uint16_t> sketch_tresholds;

            for (unsigned int i=0; i < num_repetitions; i++) {
                ConcatenatedHasher<THash> hasher(hash_family, num_bits);
                augment_result_fixed_buckets<TSim>(
                    result, hasher, points, dataset, bucket_size,
                    sketches, sketch_tresholds, use_graycodes);
            }
        }
        g_performance_metrics.store_time(Computation::Total);
        return result.get();
    }
}
