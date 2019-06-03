#include "graph.hpp"
#include "similarity_measure/cosine.hpp"

#include <iostream>
#include <fstream>

std::vector<std::vector<float>> read_glove_data(int dimensions) {
    std::string filename = "glove.6B.100d.txt";
//    std::string filename = "glove.twitter.27B.100d.txt";

    std::fstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File " << filename << " not found.";
        exit(-1);
    }
    std::vector<std::vector<float>> res;

    std::string id;
    file >> id;
    while (!file.eof()) {
        std::vector<float> row;
        float tmp;
        for (int i=0; i<dimensions; i++) {
            file >> tmp;
            row.push_back(tmp);
        }
        res.push_back(row);
        file >> id;
    }
    return res;
}

float point_recall(
    const std::vector<uint32_t>& exact,
    const std::vector<uint32_t>& approx
) {
    float correct = 0;
    for (auto neighbor : exact) {
        auto count = std::count(approx.begin(), approx.end(), neighbor);
        if (count > 0) {
            correct++;
        }
    }
    return correct/exact.size();
}

float recall(
    const std::vector<std::vector<uint32_t>>& exact,
    const std::vector<std::vector<uint32_t>>& approx
) {
    float recall_sum = 0;
    for (size_t i=0; i < exact.size(); i++) {
        recall_sum += point_recall(exact[i], approx[i]);
    }
    return recall_sum/exact.size();
}

std::vector<std::vector<uint32_t>> read_exact_result(
    std::string filename,
    size_t num_vectors,
    int k
) {
    std::fstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File " << filename << " not found.";
        exit(-1);
    }
    std::vector<std::vector<uint32_t>> res;
    for (size_t i=0; i < num_vectors; i++) {
        std::vector<uint32_t> line; 
        for (int j=0; j < k; j++) {
            uint32_t temp;
            file >> temp;
            line.push_back(temp);
        }
        res.push_back(line);
    }
    return res;
}

float len(const std::vector<float>& vec) {
    float res = 0;
    for (auto v : vec) {
        res += v*v;
    }
    return std::sqrt(res);
}

float cosine_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    float res = 0;
    for (size_t i=0; i < a.size(); i++) {
        res += a[i]*b[i];
    }
    return res/(len(a)*len(b));
}

std::vector<uint32_t> brute_force(
    const std::vector<std::vector<float>>& dataset,
    uint32_t idx,
    unsigned int k
) {
    std::vector<std::pair<uint32_t, float>> all(dataset.size());
    for (uint32_t i=0; i < dataset.size(); i++) {
        if (i != idx) {
            auto sim = cosine_similarity(dataset[idx], dataset[i]);
            all.push_back({i, sim}); 
        }
    }
    std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.second > b.second; });
    std::vector<uint32_t> res;
    for (size_t i=0; i < k; i++) {
        res.push_back(all[i].first);
    }
    return res;
}

float sample_recall(
    const std::vector<std::vector<float>>& dataset,
    const std::vector<std::vector<uint32_t>>& approx,
    unsigned int num_samples,
    unsigned int k,
    std::map<float, unsigned int>& recall_counts
) {
    std::uniform_int_distribution<uint32_t> dist(0, dataset.size()-1);
    auto& rng = puffinn::get_default_random_generator();

    float res = 0;
    for (unsigned int i=0; i < num_samples; i++) {
        auto idx = dist(rng);
        auto exact = brute_force(dataset, idx, k);
        float r = point_recall(exact, approx[idx]);
        recall_counts[r]++;
        res += r;
    }
    return res/num_samples;
}

int main() {
    const int DIMENSIONS = 100;
    const int K = 10;

    auto vectors = read_glove_data(DIMENSIONS);


    //print_optimal_hashlengths<puffinn::CosineSimilarity>(hash, dataset, K);

    auto exact = read_exact_result("glove.6B.100d.graph10.txt", vectors.size(), K);
//    auto approx = puffinn::build_graph_fixed_hash<puffinn::CosineSimilarity, std::vector<float>, puffinn::Sketch64Bit>(vectors, K, 0.7, 15, true);
    auto approx = puffinn::build_graph_fixed_buckets<puffinn::CosineSimilarity>(vectors, K, 200, 24, 64, true);
//    auto approx = puffinn::build_graph_bounded_buckets<puffinn::CosineSimilarity>(vectors, K, 200, 24, 64, true);
    /*auto approx = puffinn::build_graph_projection<puffinn::CosineSimilarity>(
        vectors, K, 100, 16, 64, true);*/
    /*auto approx = puffinn::build_graph_equal_failure_prob<puffinn::CosineSimilarity>(
        vectors, K, 0.7, 16, 0.02, false);*/
//    auto approx = puffinn::build_graph_varying_repetitions<puffinn::CosineSimilarity, std::vector<float>, puffinn::Sketch64Bit>(vectors, K, 24, 0.95, 50, true, false, 1.0);
    /*auto approx = puffinn::build_graph_nndescent<puffinn::CosineSimilarity>(vectors, K, 50, 1.0, false);*/

    std::cout << "recall: " << recall(exact, approx) << std::endl;
    std::map<float, unsigned int> recall_counts;
//    std::cout << "sampled recall: " << sample_recall(vectors, approx, 500, K, recall_counts);
    auto metrics = puffinn::g_performance_metrics.get_query_metrics()[0];
    std::cout << "\n";
    std::cout << "candidates: " << metrics.candidates << std::endl;
    std::cout << "computations: " << metrics.distance_computations << std::endl;
    std::cout << "hash lengths:";
    for (size_t i=0; i < 30; i++) {
        std::cout << " " << i << ":" << metrics.hash_length_counts[i];
    }
    std::cout << "\n";
    std::cout << "recall counts:";
    for (auto& p : recall_counts) {
        std::cout << " " << p.second << ":" << p.first;
    }
    std::cout << "\n";
    std::cout << "updates per hash length:";
    for (size_t i=0; i < 30; i++) {
        std::cout << " " << i << ":" << metrics.updates_per_hashlength[i];
    }
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "total: " << metrics.get_time(puffinn::Computation::Total) << std::endl; 
    std::cout << "setup: " << metrics.get_time(puffinn::Computation::Setup) << std::endl; 
    std::cout << "sketching: " << metrics.get_time(puffinn::Computation::Sketching) << std::endl; 
    std::cout << "hashing: " << metrics.get_time(puffinn::Computation::Hashing) << std::endl; 
    std::cout << "sketch tresholds: " << metrics.get_time(puffinn::Computation::SketchTresholdUpdate) << std::endl; 
    std::cout << "sorting: " << metrics.get_time(puffinn::Computation::Sorting) << std::endl; 
    std::cout << "augment setup: " << metrics.get_time(puffinn::Computation::AugmentSetup) << std::endl; 
    std::cout << "augment: " << metrics.get_time(puffinn::Computation::Augment) << std::endl; 

}
