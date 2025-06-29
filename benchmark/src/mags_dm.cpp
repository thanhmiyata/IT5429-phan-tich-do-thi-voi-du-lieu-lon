#include "graph_utils.h"
#include "minhash.h"
#include <omp.h>
#include <map>
#include <set>
#include <cmath>
#include <queue>

// MAGS-DM Algorithm Parameters (tương tự MAGS)
struct MAGSDMParams {
    int k;       // Số candidate pairs cho mỗi node
    int T;       // Số iterations
    int b;       // Số nodes mẫu cho 2-hop neighbors
    int h;       // Số hash functions cho MinHash
    int threads; // Số threads
    int num_partitions; // Số partitions cho divide phase
    
    MAGSDMParams() : k(5), T(50), b(5), h(10), threads(1), num_partitions(4) {}
};

struct Partition {
    std::vector<int> nodes;
    std::vector<SuperNode> local_supernodes;
    std::map<int, int> node_to_local_supernode;
    int partition_id;
    
    Partition() : partition_id(-1) {}  // Default constructor
    Partition(int id) : partition_id(id) {}
};

class MAGSDMAlgorithm {
private:
    Graph graph;
    MAGSDMParams params;
    MinHash minhash;
    std::vector<SuperNode> final_supernodes;
    std::map<int, int> node_to_supernode;
    std::vector<Partition> partitions;
    
    // Performance metrics
    double summarization_time;
    double compression_ratio;
    
public:
    MAGSDMAlgorithm(const MAGSDMParams& p) : params(p), minhash(p.h) {}
    
    bool loadGraph(const std::string& filename) {
        std::cout << "[MAGS-DM] Đang tải đồ thị từ: " << filename << std::endl;
        return graph.loadFromFile(filename);
    }
    
    void summarize() {
        GraphUtils::Timer timer;
        timer.start();
        
        std::cout << "[MAGS-DM] Bắt đầu thuật toán MAGS-DM..." << std::endl;
        std::cout << "Tham số: k=" << params.k << ", T=" << params.T 
                  << ", b=" << params.b << ", h=" << params.h 
                  << ", threads=" << params.threads 
                  << ", partitions=" << params.num_partitions << std::endl;
        
        // Phase 1: Divide - Chia đồ thị thành các partitions
        dividePhase();
        
        // Phase 2: Local Summarization - Chạy MAGS trên từng partition
        localSummarizationPhase();
        
        // Phase 3: Merge - Gộp các partitions lại
        mergePhase();
        
        summarization_time = timer.stop();
        compression_ratio = calculateCompressionRatio();
        
        std::cout << "[MAGS-DM] MAGS-DM hoàn thành!" << std::endl;
        printResults();
    }
    
    void saveSummary(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Không thể tạo file summary: " << filename << std::endl;
            return;
        }
        
        file << "# MAGS-DM Summary - " << final_supernodes.size() << " supernodes\n";
        file << "# Original: " << graph.getNumNodes() << " nodes, " << graph.getNumEdges() << " edges\n";
        file << "# Compression ratio: " << compression_ratio << "\n";
        file << "# Partitions used: " << params.num_partitions << "\n";
        
        // Ghi supernodes
        for (const auto& sn : final_supernodes) {
            if (sn.size() > 0) {
                file << "S" << sn.id << ":";
                for (int node : sn.nodes) {
                    file << " " << node;
                }
                file << "\n";
            }
        }
        
        file.close();
        std::cout << "[MAGS-DM] Summary đã được lưu vào: " << filename << std::endl;
    }
    
    // Getter methods
    double getSummarizationTime() const { return summarization_time; }
    double getCompressionRatio() const { return compression_ratio; }
    int getNumSuperNodes() const { return final_supernodes.size(); }

private:
    void dividePhase() {
        std::cout << "[MAGS-DM] Phase 1: Chia đồ thị thành " << params.num_partitions << " partitions..." << std::endl;
        
        partitions.clear();
        partitions.reserve(params.num_partitions);
        
        // Khởi tạo partitions
        for (int i = 0; i < params.num_partitions; ++i) {
            partitions.emplace_back(i);
        }
        
        // Phân chia nodes - sử dụng simple round-robin hoặc có thể dùng graph partitioning phức tạp hơn
        std::vector<int> nodes = graph.getNodes();
        for (size_t i = 0; i < nodes.size(); ++i) {
            int partition_id = i % params.num_partitions;
            partitions[partition_id].nodes.push_back(nodes[i]);
        }
        
        // In thống kê partitions
        for (int i = 0; i < params.num_partitions; ++i) {
            std::cout << "  Partition " << i << ": " << partitions[i].nodes.size() << " nodes" << std::endl;
        }
    }
    
    void localSummarizationPhase() {
        std::cout << "[MAGS-DM] Phase 2: Local summarization trên từng partition..." << std::endl;
        
        // Parallel processing của các partitions
        #pragma omp parallel for num_threads(params.threads)
        for (int p = 0; p < params.num_partitions; ++p) {
            summarizePartition(partitions[p]);
        }
        
        // In kết quả local summarization
        for (int p = 0; p < params.num_partitions; ++p) {
            int active_supernodes = 0;
            for (const auto& sn : partitions[p].local_supernodes) {
                if (sn.size() > 0) active_supernodes++;
            }
            
            std::cout << "  Partition " << p << ": " << partitions[p].nodes.size() 
                      << " nodes -> " << active_supernodes << " supernodes" << std::endl;
        }
    }
    
    void summarizePartition(Partition& partition) {
        // Tạo subgraph cho partition này
        Graph subgraph = createSubgraph(partition.nodes);
        
        // Khởi tạo local supernodes
        initializeLocalSuperNodes(partition, subgraph);
        
        // Chạy local MAGS iterations (ít iterations hơn)
        int local_iterations = std::max(1, params.T / 2);
        
        for (int iter = 0; iter < local_iterations; ++iter) {
            // Local candidate generation
            auto candidates = localCandidateGeneration(partition, subgraph);
            
            // Local greedy merge
            int merges = localGreedyMerge(partition, candidates, subgraph);
            
            if (merges == 0) break;
        }
    }
    
    Graph createSubgraph(const std::vector<int>& nodes) {
        Graph subgraph;
        std::unordered_set<int> node_set(nodes.begin(), nodes.end());
        
        // Thêm edges chỉ giữa các nodes trong partition
        for (int node : nodes) {
            const auto& neighbors = graph.getNeighbors(node);
            for (int neighbor : neighbors) {
                if (node_set.find(neighbor) != node_set.end()) {
                    subgraph.addEdge(node, neighbor);
                }
            }
        }
        
        return subgraph;
    }
    
    void initializeLocalSuperNodes(Partition& partition, const Graph& /* subgraph */) {
        partition.local_supernodes.clear();
        partition.node_to_local_supernode.clear();
        
        int sn_id = 0;
        for (int node : partition.nodes) {
            SuperNode sn(sn_id);
            sn.addNode(node);
            partition.local_supernodes.push_back(sn);
            partition.node_to_local_supernode[node] = sn_id;
            sn_id++;
        }
    }
    
    std::vector<std::pair<int, int>> localCandidateGeneration(const Partition& partition, const Graph& subgraph) {
        std::vector<std::pair<int, int>> candidates;
        
        // Simplified candidate generation cho local partition
        for (int node : partition.nodes) {
            const auto& neighbors = subgraph.getNeighbors(node);
            if (neighbors.size() < 2) continue;
            
            auto node_sig = minhash.computeSignature(neighbors);
            
            // Tìm candidates trong cùng partition
            std::vector<std::pair<double, int>> similarities;
            for (int other_node : partition.nodes) {
                if (other_node != node) {
                    const auto& other_neighbors = subgraph.getNeighbors(other_node);
                    if (other_neighbors.size() > 0) {
                        auto other_sig = minhash.computeSignature(other_neighbors);
                        double sim = minhash.jaccardSimilarity(node_sig, other_sig);
                        similarities.emplace_back(sim, other_node);
                    }
                }
            }
            
            // Sort và lấy top candidates
            std::sort(similarities.rbegin(), similarities.rend());
            int take = std::min(params.k / 2, static_cast<int>(similarities.size())); // Ít candidates hơn
            
            for (int i = 0; i < take; ++i) {
                int candidate = similarities[i].second;
                int sn1 = partition.node_to_local_supernode.at(node);
                int sn2 = partition.node_to_local_supernode.at(candidate);
                if (sn1 != sn2) {
                    candidates.emplace_back(std::min(sn1, sn2), std::max(sn1, sn2));
                }
            }
        }
        
        // Remove duplicates
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        
        return candidates;
    }
    
    int localGreedyMerge(Partition& partition, const std::vector<std::pair<int, int>>& candidates, const Graph& subgraph) {
        int merges = 0;
        
        // Simplified gain calculation cho local merge
        std::vector<std::tuple<double, int, int>> gains;
        for (const auto& pair : candidates) {
            double gain = calculateLocalMergeGain(partition, pair.first, pair.second, subgraph);
            if (gain > 0) {
                gains.emplace_back(gain, pair.first, pair.second);
            }
        }
        
        std::sort(gains.rbegin(), gains.rend());
        
        std::set<int> merged;
        for (const auto& gain_tuple : gains) {
            int sn1 = std::get<1>(gain_tuple);
            int sn2 = std::get<2>(gain_tuple);
            
            if (merged.find(sn1) == merged.end() && merged.find(sn2) == merged.end()) {
                mergeLocalSuperNodes(partition, sn1, sn2);
                merged.insert(sn1);
                merged.insert(sn2);
                merges++;
            }
        }
        
        return merges;
    }
    
    double calculateLocalMergeGain(const Partition& partition, int sn1_id, int sn2_id, const Graph& subgraph) {
        const SuperNode& sn1 = partition.local_supernodes[sn1_id];
        const SuperNode& sn2 = partition.local_supernodes[sn2_id];
        
        if (sn1.size() == 0 || sn2.size() == 0) return -1.0;
        
        int cross_edges = 0;
        for (int node1 : sn1.nodes) {
            for (int node2 : sn2.nodes) {
                if (subgraph.hasEdge(node1, node2)) {
                    cross_edges++;
                }
            }
        }
        
        return static_cast<double>(cross_edges) / (sn1.size() + sn2.size());
    }
    
    void mergeLocalSuperNodes(Partition& partition, int sn1_id, int sn2_id) {
        SuperNode& sn1 = partition.local_supernodes[sn1_id];
        SuperNode& sn2 = partition.local_supernodes[sn2_id];
        
        for (int node : sn2.nodes) {
            sn1.addNode(node);
            partition.node_to_local_supernode[node] = sn1_id;
        }
        
        sn2.nodes.clear();
    }
    
    void mergePhase() {
        std::cout << "[MAGS-DM] Phase 3: Merge partitions..." << std::endl;
        
        final_supernodes.clear();
        node_to_supernode.clear();
        
        int global_sn_id = 0;
        
        // Collect tất cả local supernodes từ các partitions
        for (const auto& partition : partitions) {
            for (const auto& local_sn : partition.local_supernodes) {
                if (local_sn.size() > 0) {
                    SuperNode global_sn(global_sn_id);
                    for (int node : local_sn.nodes) {
                        global_sn.addNode(node);
                        node_to_supernode[node] = global_sn_id;
                    }
                    final_supernodes.push_back(global_sn);
                    global_sn_id++;
                }
            }
        }
        
        // Tùy chọn: Có thể chạy thêm một vòng cross-partition merging
        crossPartitionMerging();
    }
    
    void crossPartitionMerging() {
        // Simplified cross-partition merging
        std::cout << "[MAGS-DM] Cross-partition merging..." << std::endl;
        
        // Tìm các supernodes từ khác partition có thể merge được
        std::vector<std::pair<int, int>> cross_candidates;
        
        for (size_t i = 0; i < final_supernodes.size(); ++i) {
            for (size_t j = i + 1; j < final_supernodes.size(); ++j) {
                if (final_supernodes[i].size() > 0 && final_supernodes[j].size() > 0) {
                    double similarity = calculateCrossPartitionSimilarity(i, j);
                    if (similarity > 0.5) { // Threshold
                        cross_candidates.emplace_back(i, j);
                    }
                }
            }
        }
        
        // Merge high-similarity cross-partition supernodes
        std::set<int> merged;
        int cross_merges = 0;
        for (const auto& pair : cross_candidates) {
            if (merged.find(pair.first) == merged.end() && merged.find(pair.second) == merged.end()) {
                mergeFinalSuperNodes(pair.first, pair.second);
                merged.insert(pair.first);
                merged.insert(pair.second);
                cross_merges++;
            }
        }
        
        std::cout << "  Cross-partition merges: " << cross_merges << std::endl;
    }
    
    double calculateCrossPartitionSimilarity(int sn1_id, int sn2_id) {
        const SuperNode& sn1 = final_supernodes[sn1_id];
        const SuperNode& sn2 = final_supernodes[sn2_id];
        
        // Tính Jaccard similarity dựa trên neighbors
        std::unordered_set<int> neighbors1, neighbors2;
        
        for (int node : sn1.nodes) {
            const auto& neighs = graph.getNeighbors(node);
            neighbors1.insert(neighs.begin(), neighs.end());
        }
        
        for (int node : sn2.nodes) {
            const auto& neighs = graph.getNeighbors(node);
            neighbors2.insert(neighs.begin(), neighs.end());
        }
        
        return minhash.estimateJaccard(neighbors1, neighbors2);
    }
    
    void mergeFinalSuperNodes(int sn1_id, int sn2_id) {
        SuperNode& sn1 = final_supernodes[sn1_id];
        SuperNode& sn2 = final_supernodes[sn2_id];
        
        for (int node : sn2.nodes) {
            sn1.addNode(node);
            node_to_supernode[node] = sn1_id;
        }
        
        sn2.nodes.clear();
    }
    
    double calculateCompressionRatio() {
        int active_supernodes = 0;
        for (const auto& sn : final_supernodes) {
            if (sn.size() > 0) active_supernodes++;
        }
        
        return 1.0 - static_cast<double>(active_supernodes) / graph.getNumNodes();
    }
    
    void printResults() {
        int active_supernodes = 0;
        for (const auto& sn : final_supernodes) {
            if (sn.size() > 0) active_supernodes++;
        }
        
        std::cout << "\n=== Kết quả MAGS-DM ===" << std::endl;
        std::cout << "Thời gian chạy: " << GraphUtils::Timer::formatTime(summarization_time) << std::endl;
        std::cout << "Đồ thị gốc: " << graph.getNumNodes() << " đỉnh, " << graph.getNumEdges() << " cạnh" << std::endl;
        std::cout << "Đồ thị tóm tắt: " << active_supernodes << " supernodes" << std::endl;
        std::cout << "Tỷ lệ nén: " << (compression_ratio * 100) << "%" << std::endl;
        std::cout << "Số partitions: " << params.num_partitions << std::endl;
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
    
    // Thiết lập tham số MAGS-DM
    MAGSDMParams params;
    params.threads = num_threads;
    params.num_partitions = std::min(num_threads, 8); // Số partitions phù hợp
    
    // Chạy MAGS-DM
    MAGSDMAlgorithm mags_dm(params);
    
    if (!mags_dm.loadGraph(input_path)) {
        std::cerr << "Lỗi: Không thể tải đồ thị từ " << input_path << std::endl;
        return 1;
    }
    
    mags_dm.summarize();
    mags_dm.saveSummary(output_path);
    
    // Output kết quả cho script Python (format: time_ms,compression_ratio)
    std::cout << mags_dm.getSummarizationTime() << "," << mags_dm.getCompressionRatio() << std::endl;
    
    return 0;
} 