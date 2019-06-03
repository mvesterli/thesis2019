#pragma once

#include "graph/common.hpp"

namespace puffinn {
    template <typename TSim, typename U>
    std::vector<std::vector<uint32_t>> build_graph_bf(
        const std::vector<U>& vectors,
        unsigned int k
    ) {
        if (vectors.size() == 0) {
            return {};
        }

        Dataset<typename TSim::Format> dataset(vectors[0].size(), vectors.size());
        auto dims = dataset.get_dimensions().actual;
        for (const auto& v : vectors) {
            dataset.insert(v);
        }

        Result res(vectors.size(), k);
        for (size_t i=0; i < vectors.size(); i++) {
            for (size_t j=i+1; j < vectors.size(); j++) {
                float sim = TSim::compute_similarity(dataset[i], dataset[j], dims);
                res.insert(i, j, sim);
            }
        }
        return res.get();
    }
}
