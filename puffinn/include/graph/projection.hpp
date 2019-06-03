#pragma once

#include "graph/common.hpp"
#include "format/unit_vector.hpp"

namespace puffinn {
    struct ProjectionIdx {
        uint32_t idx;
        float projected_hash;

        template <typename THash, typename U>
        static void prepare_points(
            Array<U> points,
            THash& hash_family,
            const Dataset<typename THash::Format>& dataset,
            unsigned int num_bits,
            const std::vector<float>& projection
        ) {
            g_performance_metrics.start_timer(Computation::Hashing);
            ConcatenatedHasher<THash> hasher(hash_family, num_bits);
            for (size_t i=0; i < dataset.get_size(); i++) {
                auto hash = hasher(dataset[i]);
                float projected = 0;
                for (LshDatatype bit = 0; bit < num_bits; bit++) {
                    if (hash & (1 << bit)) {
                        projected += projection[bit];
                    }
                }
                points[i].idx = i;
                points[i].projected_hash = projected;
            }
            g_performance_metrics.store_time(Computation::Hashing);
        }

        bool passes_sketch(const ProjectionIdx&) {
            return true;
        }

        template <typename U>
        static void set_sketches(
            Array<ProjectionIdx>,
            const std::vector<U>&,
            const std::vector<uint16_t>&
        ) {
        }
    };

    template <typename U>
    struct SketchedProjectionIdx {
        uint32_t idx;
        float projected_hash;
        U sketch;
        uint16_t sketch_treshold;

        bool passes_sketch(const SketchedProjectionIdx& other) {
            return passes_sketch_generic(*this, other);
        }

        static void set_sketches(
            Array<SketchedProjectionIdx> points,
            const std::vector<U>& sketches,
            const std::vector<uint16_t>& sketch_tresholds
        ) {
            SketchedIdx<U>::set_sketches(points, sketches, sketch_tresholds);
        }
    };

    template <typename TSim, typename THash, typename T, typename U>
    void augment_result_projection(
        Result& result,
        THash& hash_family,
        Array<T> points,
        const Dataset<typename TSim::Format>& dataset,
        unsigned int num_bits,
        const std::vector<float>& projection,
        unsigned int block_size,
        const std::vector<U>& sketches,
        const std::vector<uint16_t>& sketch_tresholds
    ) {
        ProjectionIdx::prepare_points(points, hash_family, dataset, num_bits, projection);
        T::set_sketches(points, sketches, sketch_tresholds);
        g_performance_metrics.start_timer(Computation::Sorting);
        std::sort(points.begin(), points.end(),
            [](T& a, T& b) {
                return a.projected_hash < b.projected_hash;
            }
        );
        g_performance_metrics.store_time(Computation::Sorting);
        g_performance_metrics.start_timer(Computation::Augment);
        for (
            size_t block_start = 0;
            block_start < points.len-(block_size-1);
            block_start += block_size
        ) {
            g_performance_metrics.add_candidates(block_size*(block_size-1)/2);
            for (size_t i=block_start; i < block_start+block_size; i++) {
                auto real_i = points[i].idx;
                for (size_t j=i+1; j < block_start+block_size; j++) {
                    if (points[i].passes_sketch(points[j])) {
                        g_performance_metrics.add_distance_computations(1);
                        auto real_j = points[j].idx;
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
    std::vector<std::vector<uint32_t>> build_graph_projection(
        const std::vector<U>& vectors,
        unsigned int k,
        unsigned int repetitions,
        unsigned int num_bits,
        unsigned int block_size,
        bool use_sketching,
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
        g_performance_metrics.store_time(Computation::Setup);

        Result result(vectors.size(), k);
        THash hash_family(dataset.get_dimensions(), dimensions, hash_args);
        if (use_sketching) {
            const unsigned int NUM_SKETCHES = 16;
            TSketch sketch_family(dataset.get_dimensions(), dimensions, sketch_args);

            std::vector<SketchedProjectionIdx<TSketchLen>> points_buffer(vectors.size());
            Array<SketchedProjectionIdx<TSketchLen>> points(points_buffer, vectors.size());
            auto sketches = generate_sketches<TSketchLen>(sketch_family, dataset, NUM_SKETCHES);
            std::vector<uint16_t> sketch_tresholds(vectors.size(), NUM_FILTER_HASHBITS);

            for (unsigned int rep=0; rep < repetitions; rep++) {
                auto projection = UnitVectorFormat::generate_random(dimensions);
                augment_result_projection<TSim>(
                    result, hash_family, points, dataset, num_bits, projection, block_size,
                    sketches[rep%NUM_SKETCHES], sketch_tresholds);
                update_sketch_tresholds(
                    sketch_family, result, sketch_tresholds, TSketchLen::length());
            }
        } else {
            std::vector<ProjectionIdx> points_buffer(vectors.size());
            Array<ProjectionIdx> points(points_buffer, vectors.size());
            std::vector<FilterLshDatatype> sketches;
            std::vector<uint16_t> sketch_tresholds;
            
            for (unsigned int rep=0; rep < repetitions; rep++) {
                auto projection = UnitVectorFormat::generate_random(dimensions);
                augment_result_projection<TSim>(
                    result, hash_family, points, dataset, num_bits, projection, block_size, sketches, sketch_tresholds);
            }
        }
        g_performance_metrics.store_time(Computation::Total);
        return result.get();
    }
}
