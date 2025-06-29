#include "graph_utils.h"
#include "minhash.h"
#include <omp.h>
#include <map>
#include <set>
#include <cmath>

// MAGS Algorithm Parameters
struct MAGSParams {
    int k;       // Số candidate pairs cho mỗi node
    int T;       // Số iterations
    int b;       // Số nodes mẫu cho 2-hop neighbors
    int h;       // Số hash functions cho MinHash
    int threads; // Số threads
    
    MAGSParams() : k(5), T(50), b(5), h(10), threads(1) {}
};

class MAGSAlgorithm {
private:
    Graph graph;
    MAGSParams params;
    MinHash minhash;
    std::vector<SuperNode> supernodes;
    std::map<int, int> node_to_supernode;
    
    // Performance metrics
    double summarization_time;
    double compression_ratio;
    
public:
    MAGSAlgorithm(const MAGSParams& p) : params(p), minhash(p.h) {}
    
    bool loadGraph(const std::string& filename) {
        std::cout << "Đang tải đồ thị từ: " << filename << std::endl;
        return graph.loadFromFile(filename);
    }
    
    void summarize() {
        GraphUtils::Timer timer;
        timer.start();
        
        std::cout << "Bắt đầu thuật toán MAGS..." << std::endl;
        std::cout << "Tham số: k=" << params.k << ", T=" << params.T 
                  << ", b=" << params.b << ", h=" << params.h 
                  << ", threads=" << params.threads << std::endl;
        
        // Khởi tạo: mỗi node là một supernode
        initializeSuperNodes();
        
        // Chạy T iterations
        for (int iter = 0; iter < params.T; ++iter) {
            std::cout << "Iteration " << (iter + 1) << "/" << params.T << "...";
            
            // Bước 1: Candidate Generation
            auto candidates = candidateGeneration();
            std::cout << " (Candidates: " << candidates.size() << ")";
            
            // Bước 2: Greedy Merge
            int merges = greedyMerge(candidates);
            std::cout << " (Merges: " << merges << ")" << std::endl;
            
            if (merges == 0) {
                std::cout << "Không có merge nào, dừng sớm tại iteration " << (iter + 1) << std::endl;
                break;
            }
        }
        
        summarization_time = timer.stop();
        compression_ratio = calculateCompressionRatio();
        
        std::cout << "MAGS hoàn thành!" << std::endl;
        printResults();
    }
    
    void saveSummary(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Không thể tạo file summary: " << filename << std::endl;
            return;
        }
        
        file << "# MAGS Summary - " << supernodes.size() << " supernodes\n";
        file << "# Original: " << graph.getNumNodes() << " nodes, " << graph.getNumEdges() << " edges\n";
        file << "# Compression ratio: " << compression_ratio << "\n";
        
        // Ghi supernodes
        for (const auto& sn : supernodes) {
            if (sn.size() > 0) {
                file << "S" << sn.id << ":";
                for (int node : sn.nodes) {
                    file << " " << node;
                }
                file << "\n";
            }
        }
        
        file.close();
        std::cout << "Summary đã được lưu vào: " << filename << std::endl;
    }
    
    // Getter methods
    double getSummarizationTime() const { return summarization_time; }
    double getCompressionRatio() const { return compression_ratio; }
    int getNumSuperNodes() const { return supernodes.size(); }

private:
    void initializeSuperNodes() {
        supernodes.clear();
        node_to_supernode.clear();
        
        int sn_id = 0;
        for (int node : graph.getNodes()) {
            SuperNode sn(sn_id);
            sn.addNode(node);
            supernodes.push_back(sn);
            node_to_supernode[node] = sn_id;
            sn_id++;
        }
        
        std::cout << "Khởi tạo " << supernodes.size() << " supernodes" << std::endl;
    }
    
    std::vector<std::pair<int, int>> candidateGeneration() {
        std::vector<std::pair<int, int>> candidates;
        std::vector<int> nodes = graph.getNodes();
        
        // Parallel candidate generation
        #pragma omp parallel for num_threads(params.threads) schedule(dynamic)
        for (size_t i = 0; i < nodes.size(); ++i) {
            int node = nodes[i];
            auto node_candidates = findCandidatesForNode(node);
            
            #pragma omp critical
            {
                candidates.insert(candidates.end(), node_candidates.begin(), node_candidates.end());
            }
        }
        
        // Remove duplicates và sort
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        
        return candidates;
    }
    
    std::vector<std::pair<int, int>> findCandidatesForNode(int node) {
        std::vector<std::pair<int, int>> candidates;
        
        // Lấy 2-hop neighbors với sampling
        auto two_hop = graph.getTwoHopNeighbors(node, params.b);
        
        if (two_hop.empty()) return candidates;
        
        // Tính MinHash signature cho node
        auto node_neighbors = graph.getNeighbors(node);
        auto node_sig = minhash.computeSignature(node_neighbors);
        
        // Tính similarity với 2-hop neighbors
        std::vector<std::pair<double, int>> similarities;
        for (int neighbor : two_hop) {
            auto neighbor_neighbors = graph.getNeighbors(neighbor);
            auto neighbor_sig = minhash.computeSignature(neighbor_neighbors);
            
            double sim = minhash.jaccardSimilarity(node_sig, neighbor_sig);
            similarities.emplace_back(sim, neighbor);
        }
        
        // Sort theo similarity và lấy top-k
        std::sort(similarities.rbegin(), similarities.rend());
        
        int take = std::min(params.k, static_cast<int>(similarities.size()));
        for (int i = 0; i < take; ++i) {
            int candidate = similarities[i].second;
            int sn1 = node_to_supernode[node];
            int sn2 = node_to_supernode[candidate];
            if (sn1 != sn2) {
                candidates.emplace_back(std::min(sn1, sn2), std::max(sn1, sn2));
            }
        }
        
        return candidates;
    }
    
    int greedyMerge(const std::vector<std::pair<int, int>>& candidates) {
        int merges = 0;
        
        // Tính gain cho mỗi candidate pair
        std::vector<std::tuple<double, int, int>> gains;
        for (const auto& pair : candidates) {
            double gain = calculateMergeGain(pair.first, pair.second);
            if (gain > 0) {
                gains.emplace_back(gain, pair.first, pair.second);
            }
        }
        
        // Sort theo gain giảm dần
        std::sort(gains.rbegin(), gains.rend());
        
        // Greedy merge
        std::set<int> merged;
        for (const auto& gain_tuple : gains) {
            int sn1 = std::get<1>(gain_tuple);
            int sn2 = std::get<2>(gain_tuple);
            
            if (merged.find(sn1) == merged.end() && merged.find(sn2) == merged.end()) {
                mergeSuperNodes(sn1, sn2);
                merged.insert(sn1);
                merged.insert(sn2);
                merges++;
            }
        }
        
        return merges;
    }
    
    double calculateMergeGain(int sn1_id, int sn2_id) {
        const SuperNode& sn1 = supernodes[sn1_id];
        const SuperNode& sn2 = supernodes[sn2_id];
        
        if (sn1.size() == 0 || sn2.size() == 0) return -1.0;
        
        // Tính số cạnh giữa hai supernodes
        int cross_edges = 0;
        for (int node1 : sn1.nodes) {
            for (int node2 : sn2.nodes) {
                if (graph.hasEdge(node1, node2)) {
                    cross_edges++;
                }
            }
        }
        
        // Gain = số cạnh tiết kiệm được
        // Simplified gain calculation
        double gain = static_cast<double>(cross_edges) / (sn1.size() + sn2.size());
        return gain;
    }
    
    void mergeSuperNodes(int sn1_id, int sn2_id) {
        SuperNode& sn1 = supernodes[sn1_id];
        SuperNode& sn2 = supernodes[sn2_id];
        
        // Merge sn2 vào sn1
        for (int node : sn2.nodes) {
            sn1.addNode(node);
            node_to_supernode[node] = sn1_id;
        }
        
        // Clear sn2
        sn2.nodes.clear();
    }
    
    double calculateCompressionRatio() {
        int active_supernodes = 0;
        for (const auto& sn : supernodes) {
            if (sn.size() > 0) active_supernodes++;
        }
        
        double ratio = 1.0 - static_cast<double>(active_supernodes) / graph.getNumNodes();
        return ratio;
    }
    
    void printResults() {
        int active_supernodes = 0;
        for (const auto& sn : supernodes) {
            if (sn.size() > 0) active_supernodes++;
        }
        
        std::cout << "\n=== Kết quả MAGS ===" << std::endl;
        std::cout << "Thời gian chạy: " << GraphUtils::Timer::formatTime(summarization_time) << std::endl;
        std::cout << "Đồ thị gốc: " << graph.getNumNodes() << " đỉnh, " << graph.getNumEdges() << " cạnh" << std::endl;
        std::cout << "Đồ thị tóm tắt: " << active_supernodes << " supernodes" << std::endl;
        std::cout << "Tỷ lệ nén: " << (compression_ratio * 100) << "%" << std::endl;
    }
};

// Main function
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Sử dụng: " << argv[0] << " <input_graph> <output_summary> <num_threads>" << std::endl;
        return 1;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int num_threads = std::stoi(argv[3]);
    
    // Thiết lập OpenMP
    omp_set_num_threads(num_threads);
    
    // Thiết lập tham số MAGS
    MAGSParams params;
    params.threads = num_threads;
    // Có thể điều chỉnh thêm tham số khác tại đây
    
    // Chạy MAGS
    MAGSAlgorithm mags(params);
    
    if (!mags.loadGraph(input_path)) {
        std::cerr << "Lỗi: Không thể tải đồ thị từ " << input_path << std::endl;
        return 1;
    }
    
    mags.summarize();
    mags.saveSummary(output_path);
    
    // Output kết quả cho script Python (format: time_ms,compression_ratio)
    std::cout << mags.getSummarizationTime() << "," << mags.getCompressionRatio() << std::endl;
    
    return 0;
} 