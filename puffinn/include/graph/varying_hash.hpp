#pragma once

#include "graph/common.hpp"

namespace puffinn {
    struct VaryingHashIdx {
        uint32_t idx;
        LshDatatype hash;
        uint8_t hash_length;

        template <typename T>
        static void prepare_points(
            Array<T> points,
            const std::vector<uint8_t>& hash_lengths
        ) {
            HashedIdx::prepare_points(points);
            for (size_t i=0; i < hash_lengths.size(); i++) {
                points[i].hash_length = hash_lengths[i];
            }
        }

        bool passes_sketch(const VaryingHashIdx&) {
            return true;
        }

        template <typename U>
        static void set_sketches(
            Array<VaryingHashIdx>,
            const std::vector<U>&,
            const std::vector<uint16_t>&
        ) {
        }
    };

    template <typename U>
    struct SketchedVaryingHashIdx {
        U sketch;
        uint32_t idx;
        LshDatatype hash;
        uint8_t hash_length;
        uint16_t sketch_treshold;

        bool passes_sketch(const SketchedVaryingHashIdx<U>& other) {
            return passes_sketch_generic(*this, other);
        }

        static void set_sketches(
            Array<SketchedVaryingHashIdx<U>> points,
            const std::vector<U>& sketches,
            const std::vector<uint16_t>& sketch_tresholds
        ) {
            SketchedIdx<U>::set_sketches(points, sketches, sketch_tresholds);
        }
    };

    template <typename TSim, typename THash, typename T, typename U>
    void augment_result_varying_hash(
        Result& result,
        ConcatenatedHasher<THash>& hasher,
        Array<T> points,
        const Dataset<typename TSim::Format>& dataset,
        const std::vector<uint8_t>& hash_lengths,
        const std::vector<U>& sketches,
        const std::vector<uint16_t>& sketch_tresholds
    ) {
        const LshDatatype ALL_MASK = 0xffffffff;

        VaryingHashIdx::prepare_points(points, hash_lengths);
        T::set_sketches(points, sketches, sketch_tresholds);
        sort_by_fixed_hash(points, hasher, dataset);

        // Pad with point outside of all other buckets as a bounds check
        // works as long as hasher.hash_length is less than 32.
        points[points.len].hash = 0xffffffff;

        g_performance_metrics.start_timer(Computation::Augment);
        for (size_t i=0; i < points.len; i++) {
            auto real_i = points[i].idx;
            auto mask = ALL_MASK << (hasher.hash_length-points[i].hash_length);
            auto prefix = points[i].hash & mask;

            // Although there are points to both sides that collide with the hash, we only look
            // forward. This is because many of the previous points have already considered this
            //  point. It does cause the guaranteed success_prob to be halved though. 
            for (size_t j=i+1; (points[j].hash & mask) == prefix; j++) {
                g_performance_metrics.add_candidates(1);
                if (points[i].passes_sketch(points[j])) {
                    g_performance_metrics.add_distance_computations(1);
                    auto real_j = points[j].idx;
                    float sim = TSim::compute_similarity(
                        dataset[real_i],
                        dataset[real_j],
                        dataset.get_dimensions().actual);
                    result.insert(real_i, real_j, sim);
                }
            }
        }
        g_performance_metrics.store_time(Computation::Augment);
    }

    template <typename T>
    void update_hash_lengths_by_success_prob(
        std::vector<uint8_t>& lengths,
        const Result& result,
        T& hasher, 
        uint8_t max_bits,
        float success_prob
    ) {
        g_performance_metrics.start_timer(Computation::AugmentSetup);
        // Only candidates ahead in the sorted list are considered, so only half the success
        // probability is guaranteed.
        success_prob *= 2;

        // Compute the length of the hash prefix necessary to get the required
        // success probability.
        float log_success_prob = std::log(success_prob);
        for (size_t i=0; i < result.size(); i++) {
            // success_prob = col_prob^hash_length ->
            // hash_length  = log(success_prob)/log(col_prob)
            float sim = result.candidates[i].smallest_value();
            float hash_length = log_success_prob/std::log(hasher.collision_probability(sim, 1));
            lengths[i] =
                std::min(max_bits, static_cast<uint8_t>(std::round(hash_length)));
        }
        g_performance_metrics.store_time(Computation::AugmentSetup);
    }

    template <
        typename TSim,
        typename U,
        typename TSketchLen = Sketch64Bit,
        typename THash = typename TSim::DefaultHash,
        typename TSketch = typename TSim::DefaultSketch
    >
    std::vector<std::vector<uint32_t>> build_graph_equal_failure_prob(
        const std::vector<U>& vectors,
        unsigned int k,
        float recall,
        unsigned int num_bits,
        float success_prob,
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

        Result result(vectors.size(), k);
        THash hash_family(dataset.get_dimensions(), dimensions, hash_args);
        float failure_prob = 1;

        std::vector<uint8_t> hash_lengths(vectors.size(), 0);
        g_performance_metrics.store_time(Computation::Setup);

        if (use_sketching) {
            const unsigned int NUM_SKETCHES = 16;
            TSketch sketch_family(dataset.get_dimensions(), dimensions, sketch_args);
            
            std::vector<SketchedVaryingHashIdx<TSketchLen>> points_buffer(vectors.size()+1);
            Array<SketchedVaryingHashIdx<TSketchLen>> points(points_buffer, vectors.size());
            auto sketches = generate_sketches<TSketchLen>(sketch_family, dataset, NUM_SKETCHES);
            std::vector<uint16_t> sketch_tresholds(vectors.size(), NUM_FILTER_HASHBITS);

            // ensure that the k'th value in the result is a reasonable starting point.
            ConcatenatedHasher<THash> hasher(hash_family, num_bits);
            augment_result_fixed_buckets<TSim>(
                result, hasher, points, dataset, k+1, 
                sketches[0], sketch_tresholds);
            for (size_t i=0; i < result.size(); i++) {
                result.candidates[i].update_minval();
            }

            size_t repetition = 0;
            while (1-failure_prob < recall) {
                ConcatenatedHasher<THash> hasher(hash_family, num_bits);
                update_hash_lengths_by_success_prob(
                    hash_lengths, result, hash_family, num_bits, success_prob);
                augment_result_varying_hash<TSim>(
                    result, hasher, points, dataset, hash_lengths,
                    sketches[repetition%NUM_SKETCHES], sketch_tresholds);
                update_sketch_tresholds(
                    sketch_family, result, sketch_tresholds, TSketchLen::length());

                failure_prob *= 1-success_prob;
                repetition++;
            }
        } else {
            std::vector<VaryingHashIdx> points_buffer(vectors.size()+1);
            Array<VaryingHashIdx> points(points_buffer, vectors.size());
            std::vector<FilterLshDatatype> sketches;
            std::vector<uint16_t> sketch_tresholds;

            // ensure that the k'th value in the result is a reasonable starting point.
            ConcatenatedHasher<THash> hasher(hash_family, num_bits);
            augment_result_fixed_buckets<TSim>(
                result, hasher, points, dataset, k+1,
                sketches, sketch_tresholds);
            for (size_t i=0; i < result.size(); i++) {
                result.candidates[i].update_minval();
            }

            while (1-failure_prob < recall) {
                ConcatenatedHasher<THash> hasher(hash_family, num_bits);
                update_hash_lengths_by_success_prob(
                    hash_lengths, result, hash_family, num_bits, success_prob);
                augment_result_varying_hash<TSim>(
                    result, hasher, points, dataset, hash_lengths,
                    sketches, sketch_tresholds);
                failure_prob *= 1-success_prob;
            }
        }
        g_performance_metrics.store_time(Computation::Total);
        return result.get();
    }
}
