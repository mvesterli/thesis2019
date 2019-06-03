#pragma once

#include <random>

namespace puffinn {
    template <typename TSim>
    void augment_result_random(
        Result& result,
        const Dataset<typename TSim::Format>& dataset
    ) {
        g_performance_metrics.start_timer(Computation::Augment);
        std::uniform_int_distribution<uint32_t> dist(0, dataset.get_size()-1);
        auto& rng = get_default_random_generator();
        for (size_t i=0; i < dataset.get_size(); i++) {
            for (unsigned int j=0; j < result.get_k(); j++) {
                auto random_idx = dist(rng); 
                auto sim = TSim::compute_similarity(
                    dataset[i],
                    dataset[random_idx],
                    dataset.get_dimensions().actual);
                result.insert(i, random_idx, sim);
            }
        }
        g_performance_metrics.store_time(Computation::Augment);
    }

    struct JoinSets {
        unsigned int set_size;
        std::vector<uint32_t> data;

        JoinSets(unsigned int n, unsigned int k, float sample_rate)
          : set_size(2+k+std::round(k*sample_rate)),
            data(n*set_size)
        {
        }

        void reset() {
            for (size_t i=0; i < data.size(); i += set_size) {
                data[i] = 0;
                data[i+1] = 0;
            }
        }

        void add(Result& result) {
            auto sampled = (set_size-2)/2;
            for (size_t i=0; i < result.size(); i++) {
                auto set = &data[i*set_size];
                for (auto j : result.candidates[i].best_indices()) {
                    auto reverse_set = &data[j*set_size];
                    set[2+set[0]] = j;
                    set[0]++;
                    if (reverse_set[1] < sampled) {
                        reverse_set[2+reverse_set[0]] = i;
                        reverse_set[0]++;
                        reverse_set[1]++;
                    }
                }
            }
        }

        uint32_t* get(size_t i) {
            return &data[i*set_size];
        }
    };

    // nn-descent algorithm based on the paper
    // 'Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures'
    template <typename TSim>
    void augment_result_nndescent(
        Result& result,
        const Dataset<typename TSim::Format>& dataset,
        // Vector of size K+pK+2, contains sets used in joins.
        // Each set contains [num_total, num_reverse, new..., old...] 
        JoinSets& join_sets
    ) {
        g_performance_metrics.start_timer(Computation::AugmentSetup);
        join_sets.reset();
        join_sets.add(result);
        g_performance_metrics.store_time(Computation::AugmentSetup);

        g_performance_metrics.start_timer(Computation::Augment);
        for (size_t i=0; i < dataset.get_size(); i++) {
            auto set = join_sets.get(i);
            auto begin = &set[2];
            auto end = &set[2+set[0]];
            for (auto v1=begin; v1 != end; v1++) {
                for (auto v2=begin; v2 != end; v2++) {
                    // There might be duplicates causing points to be compared to themselves
                    if (*v1 < *v2) { 
                        float sim = TSim::compute_similarity(
                            dataset[*v1],
                            dataset[*v2],
                            dataset.get_dimensions().actual);
                        result.insert(*v1, *v2, sim);
                    }
                }
            }
        }
        g_performance_metrics.store_time(Computation::Augment);
    }

    template <typename TSim, typename U>
    void augment_result_nndescent_with_sketching(
        Result& result,
        const Dataset<typename TSim::Format>& dataset,
        // Vector of size K+pK+2, contains sets used in joins.
        // Each set contains [num_total, num_reverse, new..., old...] 
        JoinSets& join_sets,
        const std::vector<U>& sketches,
        const std::vector<uint16_t>& sketch_tresholds
    ) {
        g_performance_metrics.start_timer(Computation::AugmentSetup);
        join_sets.reset();
        join_sets.add(result);
        g_performance_metrics.store_time(Computation::AugmentSetup);

        g_performance_metrics.start_timer(Computation::Augment);
        for (size_t i=0; i < dataset.get_size(); i++) {
            auto set = join_sets.get(i);
            auto begin = &set[2];
            auto end = &set[2+set[0]];
            for (auto v1=begin; v1 != end; v1++) {
                for (auto v2=begin; v2 != end; v2++) {
                    // There might be duplicates causing points to be compared to themselves
                    if (*v1 < *v2) { 
                        auto sketch_diff = sketches[*v1].hamming_distance(sketches[*v2]);
                        auto max_treshold = 
                            std::max(sketch_tresholds[*v1], sketch_tresholds[*v2]);
                        if (sketch_diff <= max_treshold) {
                            float sim = TSim::compute_similarity(
                                dataset[*v1],
                                dataset[*v2],
                                dataset.get_dimensions().actual);
                            result.insert(*v1, *v2, sim);
                        }
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
    std::vector<std::vector<uint32_t>> build_graph_nndescent(
        const std::vector<U>& vectors,
        unsigned int k,
        unsigned int num_repetitions,
        float sample_rate,
        bool use_sketching,
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
        augment_result_random<TSim>(result, dataset);
        g_performance_metrics.store_time(Computation::Setup);

        JoinSets join_sets(vectors.size(), k, sample_rate);

        if (use_sketching) {
            const unsigned int NUM_SKETCHES = 16;
            TSketch sketch_family(dataset.get_dimensions(), dimensions, sketch_args);
            
            auto sketches = generate_sketches<TSketchLen>(sketch_family, dataset, NUM_SKETCHES);
            std::vector<uint16_t> sketch_tresholds(vectors.size(), NUM_FILTER_HASHBITS);

            for (unsigned int i=0; i < num_repetitions; i++) {
                augment_result_nndescent_with_sketching<TSim>(
                    result, dataset, join_sets,
                    sketches[i%NUM_SKETCHES], sketch_tresholds);
                update_sketch_tresholds(
                    sketch_family, result, sketch_tresholds, TSketchLen::length());
            }
        } else {
            for (unsigned int i=0; i < num_repetitions; i++) {
                augment_result_nndescent<TSim>(result, dataset, join_sets);
            }
        }
        g_performance_metrics.store_time(Computation::Total);
        return result.get();
    }
}
