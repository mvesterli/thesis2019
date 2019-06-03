#pragma once

#include "graph/common.hpp"

namespace puffinn {
    struct FreezableHashedIdx {
        uint32_t idx;
        LshDatatype hash;
        bool frozen;

        static void set_sketches(
            Array<FreezableHashedIdx>,
            const std::vector<FilterLshDatatype>&,
            const std::vector<uint16_t>&
        ) {
        }

        static void set_individual_sketches(
            Array<FreezableHashedIdx>,
            const std::vector<FilterLshDatatype>&,
            const std::vector<uint16_t>&
        ) {
        }

        bool passes_sketch(const FreezableHashedIdx&) const {
            return true;
        }

        bool is_frozen() const {
            return frozen;
        }
    };

    template <typename T>
    struct FreezableSketchedIdx {
        uint32_t idx;
        LshDatatype hash;
        T sketch;
        uint32_t sketch_treshold;
        bool frozen;

        bool passes_sketch(const FreezableSketchedIdx<T>& other) {
            return passes_sketch_generic(*this, other);
        }

        bool is_frozen() const {
            return frozen;
        }

        template <typename U>
        static void set_sketches(
            Array<U> points,
            const std::vector<T>& sketches,
            const std::vector<uint16_t>& sketch_tresholds
        ) {
            for (size_t i=0; i < points.len; i++) {
                points[i].sketch = sketches[i];
                points[i].sketch_treshold = sketch_tresholds[i]; 
            }
        }

        static void set_individual_sketches(
            Array<FreezableSketchedIdx<T>> points,
            const std::vector<T>& sketches,
            const std::vector<uint16_t>& sketch_tresholds
        ) {
            for (size_t i=0; i < points.len; i++) {
                points[i].sketch = sketches[points[i].idx];
                points[i].sketch_treshold = sketch_tresholds[points[i].idx];
            }
        }
    };

    template <typename TSim, typename T>
    void check_bucket_with_frozen(
        Result& result,
        const Dataset<typename TSim::Format>& dataset,
        Array<T> points,
        Array<T> frozen,
        size_t cache_size
    ) {
        size_t bytes_per_value = 
            40+sizeof(T)+dataset.get_dimensions().padded*sizeof(typename TSim::Format::Type);
        cache_size = cache_size/10*9;
        size_t bucket_size = cache_size/2/bytes_per_value;

        size_t bucket_start = 0;
        while (bucket_start < points.len) {
            size_t bucket_end = std::min(bucket_start+bucket_size, points.len);

            size_t frozen_start = 0;
            while (frozen_start < frozen.len) {
                size_t frozen_end = std::min(frozen_start+bucket_size, frozen.len);

                for (size_t i=bucket_start; i < bucket_end; i++) {
                    for (size_t j=frozen_start; j < frozen_end; j++) {
                        if (points[i].passes_sketch(frozen[j])) {
                            auto sim = TSim::compute_similarity(
                                dataset[points[i].idx],
                                dataset[frozen[j].idx],
                                dataset.get_dimensions().actual);
                            g_performance_metrics.add_distance_computations(1);
                            result.insert(points[i].idx, frozen[j].idx, sim);
                        }
                    }
                }
                frozen_start = frozen_end;
            }
            bucket_start = bucket_end;
        }
    }

    // Assumes that no more work needs to be done on points. Hashes have been computed.
    template <typename TSim, typename T, bool use_frozen>
    uint64_t augment_result_fixed_hash_no_prep(
        Result& result,
        Array<T> points,
        Array<T> frozen,
        const Dataset<typename TSim::Format>& dataset,
        size_t cache_size
    ) {
        // Pad with point outside of all other buckets as a bounds check
        // works as long as num_bits is less than 32.
        points[points.len].hash = 0xffffffff;
        if (use_frozen) {
            frozen[frozen.len].hash = 0xffffffff;
        }

        g_performance_metrics.start_timer(Computation::Augment);
        uint64_t updates = 0;

        size_t bucket_start = 0;
        size_t frozen_start = 0;
        while (bucket_start < points.len) {
            size_t bucket_end = bucket_start+1;
            while (points[bucket_end].hash == points[bucket_start].hash) {
                bucket_end++;
            }

            auto bucket_size = bucket_end-bucket_start;
            auto candidate_pairs = bucket_size*(bucket_size-1)/2;
            g_performance_metrics.add_candidates(candidate_pairs);

            auto slice = points.slice(bucket_start, bucket_end);
            check_bucket<TSim>(result, dataset, slice, cache_size);
            size_t num_frozen = 0;
            for (auto& p : slice) {
                if (p.is_frozen()) {
                    num_frozen++;
                }
            }
            size_t num_live = slice.len-num_frozen;
            updates += num_live*(num_live-1)/2+num_live*num_frozen;

            if (use_frozen) {
                // Possible that whole hash buckets are skipped
                while (frozen[frozen_start].hash < points[bucket_start].hash) {
                    frozen_start++;
                }
                size_t frozen_end = frozen_start;
                while (frozen[frozen_end].hash == points[bucket_start].hash) {
                    frozen_end++;
                }
                auto frozen_size = frozen_end-frozen_start;
                g_performance_metrics.add_candidates(bucket_size*frozen_size);
                auto frozen_slice = frozen.slice(frozen_start, frozen_end);
                for (size_t i=0; i < slice.len; i++) {
                    updates += !slice[i].is_frozen()*frozen_size;
                }
                check_bucket_with_frozen<TSim>(result, dataset, slice, frozen_slice, cache_size);

                frozen_start = frozen_end;
            }
            bucket_start = bucket_end;
        }
        g_performance_metrics.store_time(Computation::Augment);
        return updates;
    }

    template <typename TSim, typename THash, typename T, typename U>
    void augment_result_fixed_hash(
        Result& result,
        ConcatenatedHasher<THash>& hasher,
        Array<T> points,
        const Dataset<typename TSim::Format>& dataset,
        const std::vector<U>& sketches,
        const std::vector<uint16_t>& sketch_tresholds,
        size_t cache_size
    ) {
        HashedIdx::prepare_points(points);
        T::set_sketches(points, sketches, sketch_tresholds);
        // Sorting instead of putting into explicit buckets.
        // Reuses code, the alternative might be faster.
        sort_by_fixed_hash(points, hasher, dataset);

        Array<T> frozen(nullptr, 0);
        augment_result_fixed_hash_no_prep<TSim, T, false>(
            result, points, frozen, dataset, cache_size);
    }

    template <typename TSim, typename THash, typename T, typename U>
    uint64_t augment_result_fixed_hash_partial_data(
        Result& result,
        ConcatenatedHasher<THash>& hasher,
        Array<T> points,
        Array<T> frozen,
        bool use_frozen,
        const Dataset<typename TSim::Format>& dataset,
        const std::vector<U>& sketches,
        const std::vector<uint16_t>& sketch_tresholds,
        size_t cache_size
    ) {
        g_performance_metrics.start_timer(Computation::Hashing);
        for (size_t i=0; i < points.len; i++) {
            points[i].hash = hasher(dataset[points[i].idx]);
        }
        if (use_frozen) {
            for (size_t i=0; i < frozen.len; i++) {
                frozen[i].hash = hasher(dataset[frozen[i].idx]);
            }
        }
        g_performance_metrics.store_time(Computation::Hashing);
        g_performance_metrics.start_timer(Computation::Sorting);
        std::sort(points.begin(), points.end(), [](T& a, T& b) { return a.hash < b.hash; });
        std::sort(frozen.begin(), frozen.end(), [](T& a, T& b) { return a.hash < b.hash; });
        g_performance_metrics.store_time(Computation::Sorting);
        T::set_individual_sketches(points, sketches, sketch_tresholds);
        T::set_individual_sketches(frozen, sketches, sketch_tresholds);

        if (use_frozen) {
            return augment_result_fixed_hash_no_prep<TSim, T, true>(
                result, points, frozen, dataset, cache_size);
        } else {
            return augment_result_fixed_hash_no_prep<TSim, T, false>(
                result, points, frozen, dataset, cache_size);
        }
    }

    template <
        typename TSim,
        typename U,
        typename TSketchLen = Sketch64Bit,
        typename THash = typename TSim::DefaultHash,
        typename TSketch = typename TSim::DefaultSketch
    >
    std::vector<std::vector<uint32_t>> build_graph_fixed_hash(
        const std::vector<U>& vectors,
        unsigned int k,
        float target_recall,
        unsigned int num_bits,
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
            
            std::vector<SketchedIdx<TSketchLen>> points(vectors.size()+1);
            auto sketches = generate_sketches<TSketchLen>(sketch_family, dataset, NUM_SKETCHES);
            std::vector<uint16_t> sketch_tresholds(vectors.size(), NUM_FILTER_HASHBITS);

            float avg_recall = 0;
            unsigned int repetition = 0;
            while (avg_recall < target_recall) {
                g_performance_metrics.add_hash_length(num_bits);
                ConcatenatedHasher<THash> hasher(hash_family, num_bits);
                augment_result_fixed_hash<TSim>(
                    result, hasher, Array<SketchedIdx<TSketchLen>>(points, vectors.size()), dataset,
                    sketches[repetition%NUM_SKETCHES], sketch_tresholds,
                    cache_size);
                update_sketch_tresholds(
                    sketch_family, result, sketch_tresholds, TSketchLen::length());
                repetition++;
                avg_recall = result.average_recall(hasher, repetition);
            }
        } else {
            std::vector<HashedIdx> points(vectors.size()+1);
            std::vector<FilterLshDatatype> sketches;
            std::vector<uint16_t> sketch_tresholds;
            float avg_recall = 0;
            unsigned int repetition = 0;
            while (avg_recall < target_recall) {
                g_performance_metrics.add_hash_length(num_bits);
                ConcatenatedHasher<THash> hasher(hash_family, num_bits);
                augment_result_fixed_hash<TSim>(
                    result, hasher, Array<HashedIdx>(points, vectors.size()), dataset,
                    sketches, sketch_tresholds, cache_size);
                repetition++;
                avg_recall = result.average_recall(hasher, repetition); 
            }
        }
        g_performance_metrics.store_time(Computation::Total);
        return result.get();
    }

    template <typename T, typename U>
    size_t mark_finished_points(
        Array<T>& points,
        const Result& result,
        ConcatenatedHasher<U>& hasher,
        const std::vector<unsigned int>& num_repetitions,
        unsigned int hash_length,
        float recall
    ) {
        g_performance_metrics.start_timer(Computation::AugmentSetup);
        size_t remaining = 0;
        for (size_t i=0; i < points.len; i++) {
            auto point = points[i];
            auto similarity = result.candidates[point.idx].smallest_value(); 
            float failure_prob = 1.0;

            for (unsigned int bits=hash_length; bits < num_repetitions.size(); bits++) {
                auto col_prob = hasher.collision_prob(similarity, bits); 
                failure_prob *= std::pow(1-col_prob, num_repetitions[bits]);
            }
            if (1-failure_prob < recall) {
                remaining++;
            } else {
                points[i].frozen = true;
            }
        }
        g_performance_metrics.store_time(Computation::AugmentSetup);
        return remaining;
    }

    template <typename T>
    void remove_frozen_points(
        Array<T>& points,
        Array<T>& frozen
    ) {
        size_t new_len = 0;
        for (size_t i=0; i < points.len; i++) {
            auto point = points[i];
            if (point.frozen) {
                frozen[frozen.len] = point;
                frozen.len++;
            } else {
                points[new_len] = point;
                new_len++;
            }
        } 
        points.len = new_len;
    }

    template <
        typename TSim,
        typename U,
        typename TSketchLen = Sketch64Bit,
        typename THash = typename TSim::DefaultHash,
        typename TSketch = typename TSim::DefaultSketch
    >
    std::vector<std::vector<uint32_t>> build_graph_varying_repetitions(
        const std::vector<U>& vectors,
        unsigned int k,
        unsigned int initial_hash_length,
        float recall,
        // Number of updates per point to maintain a hash length 
        float update_treshold,
        bool use_sketching,
        bool keep_points,
        float guaranteed_fraction = 1.0,
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

        std::vector<unsigned int> num_repetitions(initial_hash_length+1);
        g_performance_metrics.store_time(Computation::Setup);

        if (use_sketching) {
            const unsigned int NUM_SKETCHES = 16;
            TSketch sketch_family(dataset.get_dimensions(), dimensions, sketch_args);
            
            auto sketches = generate_sketches<TSketchLen>(sketch_family, dataset, NUM_SKETCHES);
            std::vector<uint16_t> sketch_tresholds(vectors.size(), NUM_FILTER_HASHBITS);

            std::vector<FreezableSketchedIdx<TSketchLen>> points_buffer(vectors.size()+1);
            std::vector<FreezableSketchedIdx<TSketchLen>> frozen_buffer(vectors.size()+1);
            Array<FreezableSketchedIdx<TSketchLen>> points(points_buffer, vectors.size());
            Array<FreezableSketchedIdx<TSketchLen>> frozen(frozen_buffer, 0);
            for (uint32_t i=0; i < points.len; i++) {
                points[i].idx = i;
                points[i].frozen = false;
            }

            unsigned int repetitions=0;
            for (
                int hash_length=initial_hash_length; 
                hash_length >= 0 && points.len > (1.0-guaranteed_fraction)*vectors.size();
                hash_length--
            ) {
                uint64_t updates = 0;
                size_t remaining = points.len;
                do {
                    g_performance_metrics.add_hash_length(hash_length);
                    ConcatenatedHasher<THash> hasher(hash_family, hash_length);

                    num_repetitions[hash_length]++;
                    updates = augment_result_fixed_hash_partial_data<TSim>(
                        result,
                        hasher,
                        points,
                        frozen,
                        keep_points,
                        dataset,
                        sketches[repetitions%NUM_SKETCHES],
                        sketch_tresholds,
                        cache_size);
                    g_performance_metrics.add_updates_per_hashlength(hash_length, updates);
                    update_sketch_tresholds(
                        sketch_family, result, sketch_tresholds, TSketchLen::length());
                    repetitions++;
                    remaining = mark_finished_points(points, result, hasher,  num_repetitions, hash_length, recall);
                } while (updates > update_treshold*remaining
                    && remaining > (1.0-guaranteed_fraction)*vectors.size());
                remove_frozen_points(points, frozen);
            }
        } else {
            std::vector<FilterLshDatatype> sketches;
            std::vector<uint16_t> sketch_tresholds;
            
            std::vector<FreezableHashedIdx> points_buffer(vectors.size()+1);
            std::vector<FreezableHashedIdx> frozen_buffer(vectors.size()+1);
            Array<FreezableHashedIdx> points(points_buffer, vectors.size());
            Array<FreezableHashedIdx> frozen(frozen_buffer, 0);
            for (uint32_t i=0; i < points.len; i++) {
                points[i].idx = i;
                points[i].frozen = false;
            }

            for (
                int hash_length=initial_hash_length;
                hash_length >= 0 && points.len > (1.0-guaranteed_fraction)*vectors.size();
                hash_length--
            ) {
                // The expected number of updates is prob(minval)*k
                uint64_t updates = 0;
                size_t remaining = points.len;
                do {
                    g_performance_metrics.add_hash_length(hash_length);
                    ConcatenatedHasher<THash> hasher(hash_family, hash_length);

                    num_repetitions[hash_length]++;
                    updates = augment_result_fixed_hash_partial_data<TSim>(
                        result,
                        hasher,
                        points,
                        frozen,
                        keep_points,
                        dataset,
                        sketches,
                        sketch_tresholds,
                        cache_size);
                    g_performance_metrics.add_updates_per_hashlength(hash_length, updates);

                    remaining = mark_finished_points(points, result, hasher, num_repetitions, hash_length, recall);
                } while (updates > update_treshold*remaining
                    && remaining > (1.0-guaranteed_fraction)*vectors.size());
                remove_frozen_points(points, frozen);
            }
        }
        g_performance_metrics.store_time(Computation::Total);
        return result.get();
    }
}
