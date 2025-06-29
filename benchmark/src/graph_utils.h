#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>

// Graph data structures
struct Edge {
    int u, v;
    Edge(int u, int v) : u(u), v(v) {}
};

struct SuperNode {
    std::unordered_set<int> nodes;
    int id;
    
    SuperNode(int id) : id(id) {}
    void addNode(int node) { nodes.insert(node); }
    size_t size() const { return nodes.size(); }
};

class Graph {
private:
    std::unordered_map<int, std::unordered_set<int>> adj_list;
    int num_nodes;
    int num_edges;
    std::vector<int> node_list;
    
public:
    Graph();
    ~Graph();
    
    // Basic operations
    void addEdge(int u, int v);
    void addNode(int node);
    bool hasEdge(int u, int v) const;
    const std::unordered_set<int>& getNeighbors(int node) const;
    
    // Graph properties
    int getNumNodes() const { return num_nodes; }
    int getNumEdges() const { return num_edges; }
    const std::vector<int>& getNodes() const { return node_list; }
    
    // I/O operations
    bool loadFromFile(const std::string& filename);
    bool saveToFile(const std::string& filename) const;
    void printStats() const;
    
    // Graph analysis
    double getDensity() const;
    int getDegree(int node) const;
    double getAverageDegree() const;
    std::vector<int> getConnectedComponents() const;
    
    // For summarization
    std::vector<int> getTwoHopNeighbors(int node, int sample_size = -1) const;
    std::vector<std::pair<int, int>> getAllEdges() const;
    
    // Iterator support
    class NodeIterator {
        const std::vector<int>& nodes;
        size_t index;
    public:
        NodeIterator(const std::vector<int>& n, size_t idx) : nodes(n), index(idx) {}
        int operator*() const { return nodes[index]; }
        NodeIterator& operator++() { ++index; return *this; }
        bool operator!=(const NodeIterator& other) const { return index != other.index; }
    };
    
    NodeIterator begin() const { return NodeIterator(node_list, 0); }
    NodeIterator end() const { return NodeIterator(node_list, node_list.size()); }
};

// Utility functions
namespace GraphUtils {
    // File format detection and conversion
    std::string detectFormat(const std::string& filename);
    bool convertToEdgeList(const std::string& input_file, const std::string& output_file);
    
    // Graph preprocessing
    Graph removeComments(const Graph& g);
    Graph makeBidirectional(const Graph& g);
    Graph removeIsolatedNodes(const Graph& g);
    
    // Performance measurement
    class Timer {
        std::chrono::high_resolution_clock::time_point start_time;
        
    public:
        void start();
        double stop(); // returns milliseconds
        static std::string formatTime(double milliseconds);
    };
    
    // Random utilities
    class RandomGenerator {
        std::mt19937 gen;
        
    public:
        RandomGenerator(unsigned seed = std::chrono::system_clock::now().time_since_epoch().count());
        int randInt(int min, int max);
        double randDouble(double min = 0.0, double max = 1.0);
        template<typename T>
        void shuffle(std::vector<T>& vec);
    };
    
    // Memory management
    size_t getMemoryUsage();
    std::string formatMemory(size_t bytes);
}

#endif // GRAPH_UTILS_H 