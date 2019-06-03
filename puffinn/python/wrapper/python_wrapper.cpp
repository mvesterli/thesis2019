#include <puffinn.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace puffinn {
namespace python {

namespace py = pybind11;

// Interface for datasets of vectors of real numbers.
// Used to have the python interface be a list of floats.
class RealLSHTable {
public:
    virtual void insert(const std::vector<float>& vec) = 0;
    virtual void rebuild(unsigned int num_threads) = 0;
    virtual std::vector<uint32_t> search_k(
        const std::vector<float>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) = 0;
};

template <typename T, typename U = SimHash>
class AngularLSHTable : public RealLSHTable {
    LSHTable<CosineSimilarity, T, U> table;

public:
    AngularLSHTable(unsigned int dimensions, uint64_t memory_limit, const HashSourceArgs<T>& hash_args)
      : table(dimensions, memory_limit, hash_args)
    {
    }

    void insert(const std::vector<float>& vec) {
        table.insert(vec);
    };
    void rebuild(unsigned int num_threads) {
        table.rebuild(num_threads);
    }
    std::vector<uint32_t> search_k(
        const std::vector<float>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search_k(vec, k, recall, filter_type);
    }
};

template <typename T>
void set(T& field, const py::dict& params, const char* name) {
    if (params.contains(name)) {
        field = py::cast<T>(params[name]);
    }
}

template <typename T, typename U = SimHash>
class EuclideanLSHTable : public RealLSHTable {
    LSHTable<L2Distance, T, U> table;

public:
    EuclideanLSHTable(unsigned int dimensions, uint64_t memory_limit, const HashSourceArgs<T>& hash_args)
      : table(dimensions, memory_limit, hash_args)
    {
    }

    void insert(const std::vector<float>& vec) {
        table.insert(vec);
    };
    void rebuild(unsigned int num_threads) {
        table.rebuild(num_threads);
    }
    std::vector<uint32_t> search_k(
        const std::vector<float>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search_k(vec, k, recall, filter_type);
    }
};

class AbstractSetLSHTable {
public:
    virtual void insert(const std::vector<uint32_t>& vec) = 0;
    virtual void rebuild(unsigned int num_threads) = 0;
    virtual std::vector<uint32_t> search_k(
        const std::vector<uint32_t>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) = 0;
};

template <typename T, typename U = MinHash1Bit>
class SetLSHTable : public AbstractSetLSHTable {
    LSHTable<JaccardSimilarity, T, U> table;

public:
    SetLSHTable(
        unsigned int dimensions,
        uint64_t memory_limit,
        const HashSourceArgs<T>& hash_args
    ) 
      : table(dimensions, memory_limit, hash_args) 
    {
    }

    void insert(const std::vector<uint32_t>& vec) {
        table.insert(vec); 
    }

    void rebuild(unsigned int num_threads) {
        table.rebuild(num_threads);
    }

    std::vector<uint32_t> search_k(
        const std::vector<uint32_t>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search_k(vec, k, recall, filter_type);
    }
};

class Index {
    std::unique_ptr<RealLSHTable> real_table;
    std::unique_ptr<AbstractSetLSHTable> set_table;
public:
    Index(
        std::string metric,
        unsigned int dimensions,
        uint64_t memory_limit,
        const py::kwargs& kwargs
    ) {
        if (metric == "angular") {
            init_angular(dimensions, memory_limit, kwargs);
        } else if (metric == "euclidean") {
            init_euclidean(dimensions, memory_limit, kwargs);
        } else if (metric == "jaccard") {
            init_jaccard(dimensions, memory_limit, kwargs);
        } else {
            throw std::invalid_argument("metric");
        }
    }

    void insert(py::list list) {
        if (real_table) {
            auto vec = list.cast<std::vector<float>>();
            real_table->insert(vec);
        } else {
            auto vec = list.cast<std::vector<unsigned int>>();
            set_table->insert(vec);
        }
    }

    void rebuild(unsigned int num_threads) {
        if (real_table) {
            real_table->rebuild(num_threads);
        } else {
            set_table->rebuild(num_threads);
        }
    }

    std::vector<uint32_t> search(
        py::list list,
        unsigned int k,
        float recall,
        std::string filter_name
    ) {
        FilterType filter_type;
        if (filter_name == "filter") {
            filter_type = FilterType::Filter;
        } else if (filter_name == "none") {
            filter_type = FilterType::None;
        } else if (filter_name == "simple") {
            filter_type = FilterType::Simple;
        } else {
            throw std::invalid_argument("filter_type");
        }
        if (real_table) {
            auto vec = list.cast<std::vector<float>>();
            return real_table->search_k(vec, k, recall, filter_type);
        } else {
            auto vec = list.cast<std::vector<unsigned int>>();
            return set_table->search_k(vec, k, recall, filter_type);
        }
    }

private:
    // No args
    void set_hash_args(SimHash::Args&, const py::dict&) {
    }

    void set_hash_args(CrossPolytopeHash::Args& args, const py::dict& params) {
        set(args.estimation_repetitions, params, "estimation_repetitions");
        set(args.estimation_eps, params, "estimation_eps");
    }

    void set_hash_args(FHTCrossPolytopeHash::Args& args, const py::dict& params) {
        set(args.estimation_eps, params, "estimation_eps");
        set(args.estimation_repetitions, params, "estimation_repetitions");
        set(args.num_rotations, params, "num_rotations");
    }

    void set_hash_args(MinHash::Args&, const py::dict&) {
    }

    template <typename T>
    std::unique_ptr<HashSourceArgs<T>> get_hash_source_args(const py::kwargs& kwargs) {
        std::string source = "pool";
        if (kwargs.contains("hash_source")) {
            source = py::cast<std::string>(kwargs["hash_source"]);
        }

        if (source == "pool") {
            unsigned int pool_size = 3000; 
            if (kwargs.contains("source_args") && kwargs["source_args"].contains("num_functions")) {
                pool_size = py::cast<unsigned int>(kwargs["source_args"]["num_functions"]);
            }
            auto res = std::make_unique<HashPoolArgs<T>>(pool_size);
            if (kwargs.contains("hash_args")) {
                set_hash_args(res->args, kwargs["hash_args"]);
            }
            return res;
        } else if (source == "independent") {
            return std::make_unique<IndependentHashArgs<T>>();
        } else if (source == "tensor") {
            return std::make_unique<TensoredHashArgs<T>>();
        } else {
            throw std::invalid_argument("hash_source");
        }
    }

    void init_angular(unsigned int dimensions, unsigned int memory_limit, const py::kwargs& kwargs) {
        std::string hash_function = "fht_crosspolytope";
        if (kwargs.contains("hash_function")) {
            hash_function = py::cast<std::string>(kwargs["hash_function"]);
        }
        if (hash_function == "simhash") {
            real_table = std::make_unique<AngularLSHTable<SimHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<SimHash>(kwargs));
        } else if (hash_function == "crosspolytope") {
            real_table = std::make_unique<AngularLSHTable<CrossPolytopeHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<CrossPolytopeHash>(kwargs));
        } else if (hash_function == "fht_crosspolytope") {
            real_table = std::make_unique<AngularLSHTable<FHTCrossPolytopeHash>>(        
                dimensions,
                memory_limit,
                *get_hash_source_args<FHTCrossPolytopeHash>(kwargs));
        } else {
            throw std::invalid_argument("hash_function");
        }
    }

    void init_euclidean(unsigned int dimensions, unsigned int memory_limit, const py::kwargs& kwargs) {
        std::string hash_function = "fht_crosspolytope";
        if (kwargs.contains("hash_function")) {
            hash_function = py::cast<std::string>(kwargs["hash_function"]);
        }
        if (hash_function == "simhash") {
            real_table = std::make_unique<EuclideanLSHTable<SimHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<SimHash>(kwargs));
        } else if (hash_function == "crosspolytope") {
            real_table = std::make_unique<EuclideanLSHTable<CrossPolytopeHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<CrossPolytopeHash>(kwargs));
        } else if (hash_function == "fht_crosspolytope") {
            real_table = std::make_unique<EuclideanLSHTable<FHTCrossPolytopeHash>>(        
                dimensions,
                memory_limit,
                *get_hash_source_args<FHTCrossPolytopeHash>(kwargs));
        } else {
            throw std::invalid_argument("hash_function");
        }
    }

    void init_jaccard(unsigned int dimensions, uint64_t memory_limit, const py::kwargs& kwargs) {
        std::string hash_function = "minhash";
        if (kwargs.contains("hash_function")) {
            hash_function = py::cast<std::string>(kwargs["hash_function"]);
        }
        if (hash_function == "minhash") {
            set_table = std::make_unique<SetLSHTable<MinHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<MinHash>(kwargs));
        } else if (hash_function == "1bit-minhash") {
            set_table = std::make_unique<SetLSHTable<MinHash1Bit>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<MinHash1Bit>(kwargs));
        } else {
            throw std::invalid_argument("hash_function");
        }
    }
};

template <typename T, typename U>
std::vector<std::vector<uint32_t>> build_graph_dispatch(
    const std::vector<std::vector<float>>& vectors,
    unsigned int k,
    const py::kwargs& kwargs
) {
    std::string method = "plain";
    std::string hash_function = "default";
    bool use_sketching = true;
    set(method, kwargs, "method");
    set(hash_function, kwargs, "hash_function");
    set(use_sketching, kwargs, "use_sketching");

    if (method == "plain") {
        float recall = 0.5;
        unsigned int hash_length = 16;
        set(recall, kwargs, "recall");
        set(hash_length, kwargs, "hash_length");

        if (hash_function == "default") {
            return build_graph_fixed_hash<T, std::vector<float>, U>(vectors, k, recall, hash_length, use_sketching);
        } else {
            throw std::invalid_argument("hash_function");
        }
    } else if (method == "fixed_buckets") {
        unsigned int repetitions = 100;
        unsigned int bucket_size = 16;
        unsigned int hash_length = 16;
        bool use_graycodes = false;
        set(repetitions, kwargs, "repetitions");
        set(bucket_size, kwargs, "bucket_size");
        set(hash_length, kwargs, "hash_length");
        set(use_graycodes, kwargs, "use_graycodes");

        if (hash_function == "default") {
            return build_graph_fixed_buckets<T, std::vector<float>, U>(
                vectors, k, repetitions, hash_length, bucket_size, use_sketching, use_graycodes);
        } else {
            throw std::invalid_argument("hash_function");
        }
    } else if (method == "bounded_buckets") {
        unsigned int repetitions = 100;
        unsigned int bucket_size = 32;
        unsigned int hash_length = 16;
        set(repetitions, kwargs, "repetitions");
        set(bucket_size, kwargs, "bucket_size");
        set(hash_length, kwargs, "hash_length");

        if (hash_function == "default") {
            return build_graph_bounded_buckets<T, std::vector<float>, U>(
                vectors, k, repetitions, hash_length, bucket_size, use_sketching);
        } else {
            throw std::invalid_argument("hash_function");
        }
    } else if (method == "varying_repetitions") {
        float recall = 0.8;
        bool keep_points = true;
        float guaranteed_fraction = 0.9;
        float update_treshold = 50.0;
        unsigned int initial_hash_length = 24;
        set(recall, kwargs, "recall");
        set(keep_points, kwargs, "keep_points");
        set(guaranteed_fraction, kwargs, "guaranteed_fraction");
        set(update_treshold, kwargs, "update_treshold");
        set(initial_hash_length, kwargs, "initial_hash_length");

        if (hash_function == "default" || hash_function == "fht_crosspolytope") {
            FHTCrossPolytopeArgs hash_args;
            if (kwargs.contains("hash_args")) {
                set(hash_args.num_rotations, kwargs["hash_args"], "num_rotations");
            }
            return build_graph_varying_repetitions<T, std::vector<float>, U, FHTCrossPolytopeHash>(
                vectors, k, initial_hash_length, recall, update_treshold, use_sketching, keep_points, guaranteed_fraction, hash_args);
        } else if (hash_function == "crosspolytope") {
            return build_graph_varying_repetitions<T, std::vector<float>, U, CrossPolytopeHash>(
                vectors, k, initial_hash_length, recall, update_treshold,
                use_sketching, keep_points, guaranteed_fraction);
        } else if (hash_function == "simhash") {
            return build_graph_varying_repetitions<T, std::vector<float>, U, SimHash>(
                vectors, k, initial_hash_length, recall, update_treshold,
                use_sketching, keep_points, guaranteed_fraction);
        } else {
            throw std::invalid_argument("hash_function");
        }
    } else if (method == "projection") {
        unsigned int repetitions = 100;
        unsigned int block_size = 16;
        unsigned int hash_length = 16;
        set(repetitions, kwargs, "repetitions");
        set(block_size, kwargs, "block_size");
        set(hash_length, kwargs, "hash_length");

        if (hash_function == "default") {
            return build_graph_projection<T, std::vector<float>, U>(
                vectors, k, repetitions, hash_length, block_size, use_sketching);
        } else {
            throw std::invalid_argument("hash_function");
        }
    } else if (method == "nndescent") {
        unsigned int repetitions = 20;
        float sample_rate = 1;
        set(repetitions, kwargs, "repetitions");
        set(sample_rate, kwargs, "sample_rate");
        if (hash_function == "default") {
            return build_graph_nndescent<T, std::vector<float>, U>(
                vectors, k, repetitions, sample_rate, use_sketching);
        } else {
            throw std::invalid_argument("hash_function");
        }
    } else {
        throw std::invalid_argument("method");
    }
}

template <typename T>
std::vector<std::vector<uint32_t>> build_graph_sketch_dispatch(
    const std::vector<std::vector<float>>& vectors,
    unsigned int k,
    const py::kwargs& kwargs
) {
    int sketch_len = 64;
    set(sketch_len, kwargs, "sketch_len");

    if (sketch_len == 64) {
        return build_graph_dispatch<T, Sketch64Bit>(vectors, k, kwargs);
    } else if (sketch_len == 128) {
        return build_graph_dispatch<T, Sketch128Bit>(vectors, k, kwargs);
    } else if (sketch_len == 256) {
        return build_graph_dispatch<T, Sketch256Bit>(vectors, k, kwargs);
    } else {
        throw std::invalid_argument("sketch_len");
    }
}

std::vector<std::vector<uint32_t>> build_graph(
    const std::string& metric,
    const std::vector<std::vector<float>>& vectors,
    unsigned int k,
    const py::kwargs& kwargs
) {
    if (metric == "angular") {
        return build_graph_sketch_dispatch<CosineSimilarity>(vectors, k, kwargs);
    } else {
        throw std::invalid_argument("metric");
    }
}

PYBIND11_MODULE(_puffinnwrapper, m) {
    py::class_<Index>(m, "Index")
        .def(py::init<const std::string&, const unsigned int&, const uint64_t&, const py::kwargs&>())
        .def("insert", &Index::insert)
        .def("rebuild", &Index::rebuild, py::arg("num_threads") = 0)
        .def("search", &Index::search,
             py::arg("vec"), py::arg("k"), py::arg("recall"),
             py::arg("filter_type") = "filter"
         );

    m.def("build_graph", build_graph);
}
} // namespace python
} // namespace puffinn
