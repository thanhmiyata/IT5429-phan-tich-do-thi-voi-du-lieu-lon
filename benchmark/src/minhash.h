#ifndef MINHASH_H
#define MINHASH_H

#include <vector>
#include <unordered_set>
#include <random>
#include <limits>
#include <functional>

class MinHash {
private:
    int num_hashes;
    std::vector<std::hash<int>> hash_functions;
    std::vector<int> a_params;  // Random parameters for hash functions
    std::vector<int> b_params;
    const int large_prime = 2147483647;  // Large prime number
    
public:
    // Constructor
    explicit MinHash(int num_hashes = 10);
    
    // Core MinHash operations
    std::vector<int> computeSignature(const std::unordered_set<int>& set) const;
    std::vector<int> computeSignature(const std::vector<int>& vec) const;
    
    // Similarity computation
    double jaccardSimilarity(const std::vector<int>& sig1, const std::vector<int>& sig2) const;
    double estimateJaccard(const std::unordered_set<int>& set1, const std::unordered_set<int>& set2) const;
    
    // Utility functions
    int getNumHashes() const { return num_hashes; }
    void printSignature(const std::vector<int>& signature) const;
    
private:
    int universalHash(int x, int a, int b) const;
    void initializeHashFunctions();
};

// MinHashLSH class for Locality Sensitive Hashing
class MinHashLSH {
private:
    MinHash minhash;
    int num_bands;
    int rows_per_band;
    std::vector<std::unordered_map<std::string, std::vector<int>>> buckets;
    
public:
    MinHashLSH(int num_hashes = 100, int num_bands = 20);
    
    // LSH operations
    void addSignature(int id, const std::vector<int>& signature);
    std::vector<int> getCandidates(const std::vector<int>& signature) const;
    
    // Utility
    void clear();
    size_t getNumBuckets() const;
    
private:
    std::string hashBand(const std::vector<int>& band) const;
};

#endif // MINHASH_H 