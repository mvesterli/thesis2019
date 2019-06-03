#pragma once 

#include "dataset.hpp"
#include "filterer.hpp"
#include "maxbuffer.hpp"
#include "typedefs.hpp"

#include <vector>

namespace puffinn {
    // Fat pointer to an array.
    // Does not copy its data when copied around.
    template <typename T>
    struct Array {
        T* data;
        size_t len;

        Array(T* data, size_t len)
          : data(data),
            len(len)
        {
        }

        Array(std::vector<T>& vec, size_t len)
          : data(&vec[0]),
            len(len)
        {
        }

        T& operator[](size_t idx) {
            return data[idx];
        }

        T* begin() {
            return data;
        }

        T* end() {
            return data+len;
        }

        Array<T> slice(size_t start, size_t end) {
            return Array(data+start, end-start);
        }
    };

    struct HashedIdx {
        uint32_t idx;
        LshDatatype hash;

        template <typename T>
        static void prepare_points(Array<T> points) {
            for (size_t i=0; i < points.len; i++) {
                points[i].idx = i;
            }
        }

        static void set_sketches(
            Array<HashedIdx>,
            const std::vector<FilterLshDatatype>&,
            const std::vector<uint16_t>&
        ) {
        }

        static void set_individual_sketches(
            Array<HashedIdx>,
            const std::vector<FilterLshDatatype>&,
            const std::vector<uint16_t>&
        ) {
        }

        bool passes_sketch(const HashedIdx&) const {
            return true;
        }

        bool is_frozen() const {
            return false;
        }
    };

    template <typename T>
    bool passes_sketch_generic(const T& a, const T& b) {
        auto sketch_diff = a.sketch.hamming_distance(b.sketch);
        auto max_treshold = std::max(a.sketch_treshold, b.sketch_treshold);
        return sketch_diff <= max_treshold;
    }

    template <typename T>
    struct ConcatenatedHasher {
        std::vector<typename T::Function> hash_functions;
        T& hash_family;
        unsigned int hash_length;

        ConcatenatedHasher(T& hash_family, unsigned int hash_length)
          : hash_family(hash_family),
            hash_length(hash_length)
        {
            auto bits_per_function = hash_family.bits_per_function();
            auto num_functions = (hash_length+bits_per_function-1)/bits_per_function; 

            for (unsigned int i=0; i < num_functions; i++) {
                hash_functions.push_back(hash_family.sample());
            }
        }

        uint64_t hash_partial(typename T::Format::Type* input, size_t function_idx) {
            return hash_functions[function_idx](input);
        }

        uint64_t operator()(typename T::Format::Type* input) {
            uint64_t res = 0;
            for (auto& f : hash_functions) {
                res <<= hash_family.bits_per_function();
                res |= f(input);
            }
            auto to_cut = hash_functions.size()*hash_family.bits_per_function()-hash_length;
            return res >> to_cut;
        }

        float collision_prob(float similarity, unsigned int bits) {
            int whole_hashes = bits/hash_family.bits_per_function();
            int remaining_bits = bits-whole_hashes*hash_family.bits_per_function();

            auto whole_prob = hash_family.collision_probability(
                similarity,
                hash_family.bits_per_function());
            auto partial_prob = hash_family.collision_probability(
                similarity,
                remaining_bits);
            return std::pow(whole_prob, whole_hashes)*partial_prob;
        }

        unsigned int partial_length() const {
            return hash_family.bits_per_function();
        }

        unsigned int get_hash_length() const {
            return hash_length;
        }

    };

    struct Sketch64Bit {
        uint64_t a;

        uint16_t hamming_distance(Sketch64Bit other) const {
            return __builtin_popcountll(a ^ other.a);
        }

        template <typename T>
        static std::vector<Sketch64Bit> generate(
            T& hash_family,
            const Dataset<typename T::Format>& dataset
        ) {
            ConcatenatedHasher<T> sketcher(hash_family, 64);
            std::vector<Sketch64Bit> res;
            res.reserve(dataset.get_size());

            for (unsigned int idx=0; idx < dataset.get_size(); idx++) {
                Sketch64Bit s;
                s.a = sketcher(dataset[idx]);
                res.push_back(s);
            }
            return res;
        }

        static uint16_t length() {
            return 64;
        }
    };

    struct Sketch128Bit {
        uint64_t a;
        uint64_t b;

        uint16_t hamming_distance(Sketch128Bit other) const {
            return __builtin_popcountll(a ^ other.a) + __builtin_popcountll(b ^ other.b);
        }

        template <typename T>
        static std::vector<Sketch128Bit> generate(
            T& hash_family,
            const Dataset<typename T::Format>& dataset
        ) {
            ConcatenatedHasher<T> sketcher_a(hash_family, 64);
            ConcatenatedHasher<T> sketcher_b(hash_family, 64);

            std::vector<Sketch128Bit> res;
            res.reserve(dataset.get_size());

            for (unsigned int idx=0; idx < dataset.get_size(); idx++) {
                Sketch128Bit s;
                s.a = sketcher_a(dataset[idx]);
                s.b = sketcher_b(dataset[idx]);
                res.push_back(s);
            }
            return res;
        }

        static uint16_t length() {
            return 128;
        }
    };

    struct Sketch256Bit {
        uint64_t a;
        uint64_t b;
        uint64_t c;
        uint64_t d;

        uint16_t hamming_distance(Sketch256Bit other) const {
            return __builtin_popcountll(a ^ other.a)
                + __builtin_popcountll(b ^ other.b)
                + __builtin_popcountll(c ^ other.c)
                + __builtin_popcountll(d ^ other.d);
        }

        template <typename T>
        static std::vector<Sketch256Bit> generate(
            T& hash_family,
            const Dataset<typename T::Format>& dataset
        ) {
            ConcatenatedHasher<T> sketcher_a(hash_family, 64);
            ConcatenatedHasher<T> sketcher_b(hash_family, 64);
            ConcatenatedHasher<T> sketcher_c(hash_family, 64);
            ConcatenatedHasher<T> sketcher_d(hash_family, 64);

            std::vector<Sketch256Bit> res;
            res.reserve(dataset.get_size());

            for (unsigned int idx=0; idx < dataset.get_size(); idx++) {
                Sketch256Bit s;
                s.a = sketcher_a(dataset[idx]);
                s.b = sketcher_b(dataset[idx]);
                s.c = sketcher_c(dataset[idx]);
                s.d = sketcher_d(dataset[idx]);
                res.push_back(s);
            }
            return res;
        }

        static uint16_t length() {
            return 256;
        }
    };

    template <typename T>
    struct SketchedIdx {
        uint32_t idx;
        LshDatatype hash;
        T sketch;
        uint32_t sketch_treshold;

        bool passes_sketch(const SketchedIdx<T>& other) {
            return passes_sketch_generic(*this, other);
        }

        bool is_frozen() const {
            return false;
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
            Array<SketchedIdx<T>> points,
            const std::vector<T>& sketches,
            const std::vector<uint16_t>& sketch_tresholds
        ) {
            for (size_t i=0; i < points.len; i++) {
                points[i].sketch = sketches[points[i].idx];
                points[i].sketch_treshold = sketch_tresholds[points[i].idx];
            }
        }
    };

    // The combined results for each point in the dataset.
    struct Result {
        unsigned int k;
        std::vector<MaxBuffer> candidates;

        Result(size_t size, unsigned int k)
          : k(k)
        {
            candidates.reserve(size);
            for (size_t i=0; i < size; i++) {
                candidates.emplace_back(k);
            }
        }

        // Insert the symmetric distance between i and j.
        bool insert(
            uint_fast32_t i,
            uint_fast32_t j,
            float similarity
        ) {
            auto a = candidates[i].insert(j, similarity);
            auto b = candidates[j].insert(i, similarity);
            return a || b;
        }

        unsigned int get_k() const {
            return k;
        }

        size_t size() const {
            return candidates.size(); 
        }

        std::vector<std::vector<uint32_t>> get() {
            std::vector<std::vector<uint32_t>> results;
            for (auto& c : candidates) {
                results.push_back(c.best_indices());
            }
            return results;
        }

        template <typename T>
        float average_recall(
            ConcatenatedHasher<T>& hasher,
            unsigned int num_repetitions
        ) {
            g_performance_metrics.start_timer(Computation::AugmentSetup);
            float sum = 0;
            for (auto& c : candidates) {
                float sim = c.smallest_value();
                float col_prob = hasher.collision_prob(sim, hasher.hash_length);
                sum += 1.0-std::pow(1.0-col_prob, num_repetitions);
            }
            g_performance_metrics.store_time(Computation::AugmentSetup);
            return sum/size();
        }
    };

    template <typename T, typename THash>
    void compute_fixed_hashes(
        Array<T> points,
        ConcatenatedHasher<THash>& hash,
        const Dataset<typename THash::Format>& dataset
    ) {
        g_performance_metrics.start_timer(Computation::Hashing);
        for (size_t i=0; i < points.len; i++) {
            points[i].hash = hash(dataset[i]);
        }
        g_performance_metrics.store_time(Computation::Hashing);
    }

    // TODO try hash using minimal number of hash computations.
    template <typename T, typename THash>
    void sort_by_fixed_hash(
        Array<T> points,
        ConcatenatedHasher<THash>& hasher,
        const Dataset<typename THash::Format>& dataset
    ) {
        compute_fixed_hashes(points, hasher, dataset);

        g_performance_metrics.start_timer(Computation::Sorting);
        std::sort(points.begin(), points.end(), [](T& a, T& b) {
            return a.hash < b.hash;
        });
        g_performance_metrics.store_time(Computation::Sorting);
    }

    template <typename T>
    void update_sketch_tresholds(
        T& hasher,
        const Result& result,
        std::vector<uint16_t>& tresholds,
        uint16_t sketch_len 
    ) {
        g_performance_metrics.start_timer(Computation::SketchTresholdUpdate);
        for (size_t i=0; i < tresholds.size(); i++) {
            auto minval = result.candidates[i].smallest_value();
            float col_prob = hasher.collision_probability(minval, 1);
            tresholds[i] = sketch_len*(1.0-col_prob);
        }
        g_performance_metrics.store_time(Computation::SketchTresholdUpdate);
    }

    template <typename U, typename T>
    std::vector<std::vector<U>> generate_sketches(
        T& hash_family,
        const Dataset<typename T::Format>& dataset,
        unsigned int num_sketches
    ) {
        g_performance_metrics.start_timer(Computation::Sketching);
        std::vector<std::vector<U>> res;
        for (unsigned int sketch=0; sketch < num_sketches; sketch++) {
            res.push_back(U::generate(hash_family, dataset));
        }
        g_performance_metrics.store_time(Computation::Sketching);
        return res;
    }

    uint8_t prefix_length(LshDatatype a, LshDatatype b, unsigned int num_bits) {
        uint8_t res = 0;
        LshDatatype mask = (1 << (num_bits-1));
        while (mask != 0 && (a & mask) == (b & mask)) {
            res++;
            mask >>= 1;
        }
        return res;
    }

    template <typename TSim, typename T>
    void check_bucket(
        Result& result,
        const Dataset<typename TSim::Format>& dataset,
        Array<T> points,
        size_t cache_size
    ) {
        // num bytes:
        // 40*bucket size from MaxBuffer
        // sizeof(T)*bucket size
        // padded_dimensions*sizeof(formattype)*bucket size 
        size_t bytes_per_value = 
            40+sizeof(T)+dataset.get_dimensions().padded*sizeof(typename TSim::Format::Type);
        cache_size *= 0.9; // to be safe.
        size_t bucket_size = cache_size/bytes_per_value;
        // Ensure even
        if (bucket_size % 2 == 1) {
            bucket_size--;
        }

        // Inner joins
        size_t bucket_start = 0;
        while (bucket_start < points.len) {
            size_t bucket_end = std::min(bucket_start+bucket_size, points.len);
            for (size_t i=bucket_start; i < bucket_end; i++) {
                for (size_t j=i+1; j < bucket_end; j++) {
                    if (points[i].passes_sketch(points[j])) {
                        auto sim = TSim::compute_similarity(
                            dataset[points[i].idx],
                            dataset[points[j].idx],
                            dataset.get_dimensions().actual);
                        g_performance_metrics.add_distance_computations(1);
                        result.insert(points[i].idx, points[j].idx, sim);
                    }
                }
            }
            bucket_start = bucket_end;
        }

        // Outer joins
        size_t bucket_a_start = 0;
        int iteration = 0;
        while (bucket_a_start < points.len) {
            size_t bucket_a_end = std::min(bucket_a_start+bucket_size/2, points.len);
            size_t bucket_b_start = iteration == 0
                ? bucket_a_start+bucket_size
                : bucket_a_start+bucket_size/2;

            while (bucket_b_start < points.len) {
                size_t bucket_b_end = std::min(bucket_b_start+bucket_size/2, points.len);
                for (size_t i=bucket_a_start; i < bucket_a_end; i++) {
                    for (size_t j=bucket_b_start; j < bucket_b_end; j++) {
                        if (points[i].passes_sketch(points[j])) {
                            auto sim = TSim::compute_similarity(
                                dataset[points[i].idx],
                                dataset[points[j].idx],
                                dataset.get_dimensions().actual);
                            g_performance_metrics.add_distance_computations(1);
                            result.insert(points[i].idx, points[j].idx, sim);
                        }
                    }
                }
                bucket_b_start = bucket_b_end;
            }

            bucket_a_start = bucket_a_end;
            iteration++;
        }
    }
}
