#pragma once

#include "typedefs.hpp"
#include "performance.hpp"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace puffinn {
    // Stores `k` indices and associated values, so that the top indices can easily be retrieved.
    class MaxBuffer {
    public:
        using ResultPair = std::pair<uint32_t, float>;

    private:
        const unsigned int size;
        unsigned int inserted_values;
        float minval;
        std::vector<ResultPair> data;

        // Reorder the values, so that the top `k` elements are stored first.
        // All other values are removed.
        void filter() {
            g_performance_metrics.start_timer(Computation::MaxbufferFilter);
            std::sort(data.begin(), data.begin()+inserted_values,
                [](const ResultPair& a, const ResultPair& b) {
                    return a.second > b.second
                        || (a.second == b.second && a.first > b.first);
                });

            // Deduplication step
            unsigned int deduplicated_values = std::min(1u, inserted_values);
            for (unsigned int idx=1; idx < inserted_values; idx++) {
                if (data[idx].first != data[deduplicated_values-1].first) {
                    data[deduplicated_values] = data[idx];
                    deduplicated_values++;
                }
            }
            inserted_values = std::min(deduplicated_values, size);
            if (inserted_values != 0) {
                if (inserted_values == size) {
                    minval = data[inserted_values-1].second;
                } else {
                    minval = 0.0;
                }
            }
            g_performance_metrics.store_time(Computation::MaxbufferFilter);
        }

    public:
        // Construct a buffer containing `size` elements. The memory used is twice that.
        MaxBuffer(unsigned int k)
          : size(k),
            inserted_values(0),
            minval(0.0),
            data(std::vector<ResultPair>(2*k))
        {
            if (k == 0) {
                // Make it impossible to insert.
                minval = 1.0;
            }
        }

        // Insert an index with an associated value into the buffer.
        // The buffer may choose to ignore it if it is not relevant.
        bool insert(uint32_t idx, float value) {
            if (value <= minval) { return false; } // value is not relevant

            if (inserted_values == 2*size) {
                filter();
            }
            data[inserted_values] = { idx, value };
            inserted_values++;
            return true;
        }

        // Retrieve the `k` entries with the highest associated values.
        std::vector<ResultPair> best_entries() {
            filter();
            std::vector<ResultPair> res;
            for (unsigned i=0; i<inserted_values; i++) {
                res.push_back(data[i]);
            }
            return res;
        }

        std::vector<uint32_t> best_indices() {
            auto entries = best_entries();
            std::vector<uint32_t> res;
            res.reserve(entries.size());
            for (auto entry : entries) {
                res.push_back(entry.first);
            }
            return res;
        }

        // Retrieve the current smallest values that inserted values have to beat
        // in order to be considered.
        float smallest_value() const {
            return minval;
        }

        // Force filtering the buffer.
        void update_minval() {
            filter();
        }

        // Retrieve the i'th inserted index when sorted by their descending values.
        //
        // buffer[0] will be the index with the largest associated value,
        // buffer[k-1] will be the index with the least associated value.
        // This does not take into account values that are inserted after the last filter.
        uint32_t operator[](size_t i) const {
            return data[i].first;
        }
    };
}
