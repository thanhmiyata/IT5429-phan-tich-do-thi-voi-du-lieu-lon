#include "graph_utils.h"
#include <cstdlib>
#include <climits>
#include <queue>

// Graph class implementation
Graph::Graph() : num_nodes(0), num_edges(0) {}

Graph::~Graph() {}

void Graph::addEdge(int u, int v) {
    if (adj_list[u].find(v) == adj_list[u].end()) {
        adj_list[u].insert(v);
        num_edges++;
    }
    addNode(u);
    addNode(v);
}

void Graph::addNode(int node) {
    if (adj_list.find(node) == adj_list.end()) {
        adj_list[node] = std::unordered_set<int>();
        node_list.push_back(node);
        num_nodes++;
    }
}

bool Graph::hasEdge(int u, int v) const {
    auto it = adj_list.find(u);
    if (it != adj_list.end()) {
        return it->second.find(v) != it->second.end();
    }
    return false;
}

const std::unordered_set<int>& Graph::getNeighbors(int node) const {
    static std::unordered_set<int> empty_set;
    auto it = adj_list.find(node);
    return (it != adj_list.end()) ? it->second : empty_set;
}

bool Graph::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Không thể mở file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int edges_loaded = 0;
    
    while (std::getline(file, line)) {
        // Bỏ qua các dòng comment
        if (line.empty() || line[0] == '#' || line[0] == '%') {
            continue;
        }
        
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            addEdge(u, v);
            edges_loaded++;
            
            if (edges_loaded % 100000 == 0) {
                std::cout << "Đã tải " << edges_loaded << " cạnh...\r" << std::flush;
            }
        }
    }
    
    std::cout << "\nTải xong: " << num_nodes << " đỉnh, " << num_edges << " cạnh" << std::endl;
    file.close();
    return true;
}

bool Graph::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# Graph với " << num_nodes << " đỉnh và " << num_edges << " cạnh\n";
    for (const auto& node_pair : adj_list) {
        for (int neighbor : node_pair.second) {
            file << node_pair.first << " " << neighbor << "\n";
        }
    }
    
    file.close();
    return true;
}

void Graph::printStats() const {
    std::cout << "=== Thống kê đồ thị ===" << std::endl;
    std::cout << "Số đỉnh: " << num_nodes << std::endl;
    std::cout << "Số cạnh: " << num_edges << std::endl;
    std::cout << "Mật độ: " << getDensity() << std::endl;
    std::cout << "Bậc trung bình: " << getAverageDegree() << std::endl;
}

double Graph::getDensity() const {
    if (num_nodes <= 1) return 0.0;
    return static_cast<double>(num_edges) / (static_cast<double>(num_nodes) * (num_nodes - 1));
}

int Graph::getDegree(int node) const {
    auto it = adj_list.find(node);
    return (it != adj_list.end()) ? it->second.size() : 0;
}

double Graph::getAverageDegree() const {
    if (num_nodes == 0) return 0.0;
    
    int total_degree = 0;
    for (const auto& node_pair : adj_list) {
        total_degree += node_pair.second.size();
    }
    
    return static_cast<double>(total_degree) / num_nodes;
}

std::vector<int> Graph::getTwoHopNeighbors(int node, int sample_size) const {
    std::unordered_set<int> two_hop_set;
    
    // Lấy 1-hop neighbors
    const auto& one_hop = getNeighbors(node);
    
    // Lấy 2-hop neighbors
    for (int neighbor : one_hop) {
        const auto& neighbor_neighbors = getNeighbors(neighbor);
        for (int two_hop : neighbor_neighbors) {
            if (two_hop != node && one_hop.find(two_hop) == one_hop.end()) {
                two_hop_set.insert(two_hop);
            }
        }
    }
    
    std::vector<int> result(two_hop_set.begin(), two_hop_set.end());
    
    // Sampling nếu cần
    if (sample_size > 0 && result.size() > static_cast<size_t>(sample_size)) {
        GraphUtils::RandomGenerator rng;
        rng.shuffle(result);
        result.resize(sample_size);
    }
    
    return result;
}

std::vector<std::pair<int, int>> Graph::getAllEdges() const {
    std::vector<std::pair<int, int>> edges;
    edges.reserve(num_edges);
    
    for (const auto& node_pair : adj_list) {
        for (int neighbor : node_pair.second) {
            edges.emplace_back(node_pair.first, neighbor);
        }
    }
    
    return edges;
}

// GraphUtils namespace implementation
namespace GraphUtils {
    
    std::string detectFormat(const std::string& filename) {
        if (filename.find(".txt") != std::string::npos) return "edge_list";
        if (filename.find(".gz") != std::string::npos) return "compressed";
        return "unknown";
    }
    
    void Timer::start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double Timer::stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        return static_cast<double>(duration.count());
    }
    
    std::string Timer::formatTime(double milliseconds) {
        if (milliseconds < 1000) {
            return std::to_string(static_cast<int>(milliseconds)) + "ms";
        } else if (milliseconds < 60000) {
            return std::to_string(milliseconds / 1000.0) + "s";
        } else {
            int minutes = static_cast<int>(milliseconds / 60000);
            int seconds = static_cast<int>((milliseconds - minutes * 60000) / 1000);
            return std::to_string(minutes) + "m" + std::to_string(seconds) + "s";
        }
    }
    
    RandomGenerator::RandomGenerator(unsigned seed) : gen(seed) {}
    
    int RandomGenerator::randInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }
    
    double RandomGenerator::randDouble(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen);
    }
    
    template<typename T>
    void RandomGenerator::shuffle(std::vector<T>& vec) {
        std::shuffle(vec.begin(), vec.end(), gen);
    }
    
    // Explicit template instantiation
    template void RandomGenerator::shuffle<int>(std::vector<int>&);
    template void RandomGenerator::shuffle<std::pair<int, int>>(std::vector<std::pair<int, int>>&);
    
    size_t getMemoryUsage() {
        // Platform-specific memory usage
        #ifdef __APPLE__
            // macOS implementation would go here
            return 0;
        #elif __linux__
            // Linux implementation would go here
            return 0;
        #else
            return 0;
        #endif
    }
    
    std::string formatMemory(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit = 0;
        double size = static_cast<double>(bytes);
        
        while (size >= 1024.0 && unit < 4) {
            size /= 1024.0;
            unit++;
        }
        
        char buffer[64];
        std::sprintf(buffer, "%.2f%s", size, units[unit]);
        return std::string(buffer);
    }
} 