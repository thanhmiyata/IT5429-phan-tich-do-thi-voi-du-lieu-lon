#include "minhash.h"
#include <iostream>
#include <sstream>
#include <algorithm>

// MinHash implementation
MinHash::MinHash(int num_hashes) : num_hashes(num_hashes) {
    hash_functions.resize(num_hashes);
    a_params.resize(num_hashes);
    b_params.resize(num_hashes);
    initializeHashFunctions();
}

void MinHash::initializeHashFunctions() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, large_prime - 1);
    
    for (int i = 0; i < num_hashes; ++i) {
        a_params[i] = dis(gen);
        b_params[i] = dis(gen);
    }
}

int MinHash::universalHash(int x, int a, int b) const {
    // Universal hash function: ((a*x + b) mod p) mod large_number
    long long result = (static_cast<long long>(a) * x + b) % large_prime;
    return static_cast<int>(result);
}

std::vector<int> MinHash::computeSignature(const std::unordered_set<int>& set) const {
    std::vector<int> signature(num_hashes, std::numeric_limits<int>::max());
    
    for (int element : set) {
        for (int i = 0; i < num_hashes; ++i) {
            int hash_val = universalHash(element, a_params[i], b_params[i]);
            signature[i] = std::min(signature[i], hash_val);
        }
    }
    
    return signature;
}

std::vector<int> MinHash::computeSignature(const std::vector<int>& vec) const {
    std::unordered_set<int> set(vec.begin(), vec.end());
    return computeSignature(set);
}

double MinHash::jaccardSimilarity(const std::vector<int>& sig1, const std::vector<int>& sig2) const {
    if (sig1.size() != sig2.size()) {
        std::cerr << "Lỗi: Signatures có kích thước khác nhau!" << std::endl;
        return 0.0;
    }
    
    int matches = 0;
    for (size_t i = 0; i < sig1.size(); ++i) {
        if (sig1[i] == sig2[i]) {
            matches++;
        }
    }
    
    return static_cast<double>(matches) / sig1.size();
}

double MinHash::estimateJaccard(const std::unordered_set<int>& set1, const std::unordered_set<int>& set2) const {
    auto sig1 = computeSignature(set1);
    auto sig2 = computeSignature(set2);
    return jaccardSimilarity(sig1, sig2);
}

void MinHash::printSignature(const std::vector<int>& signature) const {
    std::cout << "[";
    for (size_t i = 0; i < signature.size(); ++i) {
        std::cout << signature[i];
        if (i < signature.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// MinHashLSH implementation
MinHashLSH::MinHashLSH(int num_hashes, int num_bands) 
    : minhash(num_hashes), num_bands(num_bands) {
    
    if (num_hashes % num_bands != 0) {
        std::cerr << "Cảnh báo: num_hashes không chia hết cho num_bands" << std::endl;
    }
    
    rows_per_band = num_hashes / num_bands;
    buckets.resize(num_bands);
}

void MinHashLSH::addSignature(int id, const std::vector<int>& signature) {
    for (int band = 0; band < num_bands; ++band) {
        int start_idx = band * rows_per_band;
        int end_idx = std::min(start_idx + rows_per_band, static_cast<int>(signature.size()));
        
        std::vector<int> band_signature(signature.begin() + start_idx, signature.begin() + end_idx);
        std::string band_hash = hashBand(band_signature);
        
        buckets[band][band_hash].push_back(id);
    }
}

std::vector<int> MinHashLSH::getCandidates(const std::vector<int>& signature) const {
    std::unordered_set<int> candidate_set;
    
    for (int band = 0; band < num_bands; ++band) {
        int start_idx = band * rows_per_band;
        int end_idx = std::min(start_idx + rows_per_band, static_cast<int>(signature.size()));
        
        std::vector<int> band_signature(signature.begin() + start_idx, signature.begin() + end_idx);
        std::string band_hash = hashBand(band_signature);
        
        auto bucket_it = buckets[band].find(band_hash);
        if (bucket_it != buckets[band].end()) {
            for (int candidate : bucket_it->second) {
                candidate_set.insert(candidate);
            }
        }
    }
    
    return std::vector<int>(candidate_set.begin(), candidate_set.end());
}

void MinHashLSH::clear() {
    for (auto& bucket : buckets) {
        bucket.clear();
    }
}

size_t MinHashLSH::getNumBuckets() const {
    size_t total = 0;
    for (const auto& bucket : buckets) {
        total += bucket.size();
    }
    return total;
}

std::string MinHashLSH::hashBand(const std::vector<int>& band) const {
    std::ostringstream oss;
    for (size_t i = 0; i < band.size(); ++i) {
        oss << band[i];
        if (i < band.size() - 1) oss << "_";
    }
    return oss.str();
} 