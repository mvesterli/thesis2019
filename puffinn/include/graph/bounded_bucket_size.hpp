#pragma once

#include "graph/common.hpp"

namespace puffinn {
    template <typename TSim, typename T, typename THash>
    void check_buckets(
        Result& result,
        Array<T> points,
        ConcatenatedHasher<THash> hasher,
        const Dataset<typename TSim::Format>& dataset,
        size_t depth,
        uint32_t max_bucket_size,
        size_t cache_size
    ) {
        if (points.len <= max_bucket_size || depth >= hasher.get_hash_length()) {
            check_bucket<TSim>(result, dataset, points, cache_size);
        } else {
            if (depth % hasher.partial_length() == 0) {
                for (size_t i=0; i < points.len; i++) {
                    points[i].hash = hasher.hash_partial(dataset[points[i].idx], depth/hasher.partial_length());
                }
            }
            LshDatatype mask = 1 << (hasher.partial_length()-1-depth % hasher.partial_length());
            size_t mid = points.len-1;
            size_t i=0;
            while (i < mid) {
                // put zeros first
                if (points[i].hash & mask) {
                    std::swap(points[mid], points[i]);
                    mid--;
                } else {
                    i++;
                }
            }
            Array<T> zeros(&points[0], mid);
            Array<T> ones(&points[mid], points.len-mid);
            check_buckets<TSim>(result, zeros, hasher, dataset, depth+1, max_bucket_size, cache_size);
            check_buckets<TSim>(result, ones, hasher, dataset, depth+1, max_bucket_size, cache_size);
        }
    }
    
    template <typename TSim, typename THash, typename TPoint, typename TSketch>
    void augment_result_bounded_buckets(
        Result& result,
        ConcatenatedHasher<THash>& hasher,
        Array<TPoint> points,
        const Dataset<typename TSim::Format>& dataset,
        unsigned int bucket_size,
        const std::vector<TSketch>& sketches,
        const std::vector<uint16_t>& sketch_tresholds,
        size_t cache_size
    ) {
        HashedIdx::prepare_points(points);
        TPoint::set_sketches(points, sketches, sketch_tresholds);

        g_performance_metrics.start_timer(Computation::Augment);
        check_buckets<TSim>(result, points, hasher, dataset, 0, bucket_size, cache_size);
        g_performance_metrics.store_time(Computation::Augment);
    }

    template <
        typename TSim,
        typename U,
        typename TSketchLen = Sketch64Bit,
        typename THash = typename TSim::DefaultHash,
        typename TSketch = typename TSim::DefaultSketch
    >
    std::vector<std::vector<uint32_t>> build_graph_bounded_buckets(
        const std::vector<U>& vectors,
        unsigned int k,
        unsigned int num_repetitions,
        unsigned int num_bits,
        unsigned int bucket_size,
        bool use_sketching,
        typename THash::Args hash_args = typename THash::Args(),
        typename TSketch::Args sketch_args = typename TSketch::Args(),
        size_t cache_size = 32*1024
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
                augment_result_bounded_buckets<TSim>(
                    result, hasher, points, dataset, bucket_size,
                    sketches[i%NUM_SKETCHES], sketch_tresholds, cache_size);
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
                augment_result_bounded_buckets<TSim>(
                    result, hasher, points, dataset, bucket_size,
                    sketches, sketch_tresholds, cache_size);
            }
        }
        g_performance_metrics.store_time(Computation::Total);
        return result.get();
    }
}
