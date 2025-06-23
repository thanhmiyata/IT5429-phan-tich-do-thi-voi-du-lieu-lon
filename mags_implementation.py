#!/usr/bin/env python3
"""
Triển khai cơ bản của thuật toán MAGS (Multi-phase Algorithms for Graph Summarization)
Dựa trên paper SIGMOD 2024: "Graph Summarization: Compactness Meets Efficiency"
"""

import numpy as np
import networkx as nx
from collections import defaultdict, deque
import random
import math
import time
from typing import Dict, List, Tuple, Set, Optional
import hashlib

class SuperNode:
    """Lớp đại diện cho super-node trong summary graph"""
    def __init__(self, node_id: int, nodes: Set[int]):
        self.id = node_id
        self.nodes = nodes
        self.size = len(nodes)
        self.neighbors = set()
    
    def __str__(self):
        return f"SuperNode({self.id}, size={self.size}, nodes={self.nodes})"

class MAGS:
    """
    Triển khai thuật toán MAGS cho graph summarization
    """
    
    def __init__(self, k: int = 5, T: int = 50, b: int = 5, h: int = 10):
        """
        Khởi tạo MAGS
        
        Args:
            k: Số lượng candidate pairs cho mỗi node
            T: Số vòng lặp
            b: Số nodes mẫu cho 2-hop neighbors
            h: Số hash functions cho MinHash
        """
        self.k = k
        self.T = T  
        self.b = b
        self.h = h
        self.candidate_pairs = []
        self.super_nodes = {}
        self.super_edges = {}
        
    def _compute_minhash(self, graph: nx.Graph, node: int) -> List[int]:
        """Tính MinHash signatures cho một node"""
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return [0] * self.h
            
        signatures = []
        for i in range(self.h):
            # Sử dụng hash function đơn giản
            min_hash = min(hash(f"{neighbor}_{i}") % (2**31) for neighbor in neighbors)
            signatures.append(min_hash)
        return signatures
    
    def _jaccard_similarity(self, graph: nx.Graph, u: int, v: int) -> float:
        """Tính Jaccard similarity giữa hai nodes"""
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        
        if not neighbors_u and not neighbors_v:
            return 1.0
        if not neighbors_u or not neighbors_v:
            return 0.0
            
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        return intersection / union if union > 0 else 0.0
    
    def _generate_candidate_pairs(self, graph: nx.Graph) -> List[Tuple[int, int, float]]:
        """
        Phase 1: Candidate Generation
        Tạo các cặp candidate nodes dựa trên MinHash và similarity
        """
        print("Generating candidate pairs...")
        candidates = []
        nodes = list(graph.nodes())
        
        # Tính MinHash cho tất cả nodes
        minhash_signatures = {}
        for node in nodes:
            minhash_signatures[node] = self._compute_minhash(graph, node)
        
        # Tìm các cặp candidates dựa trên similarity
        for i, u in enumerate(nodes):
            similarities = []
            
            # Lấy 2-hop neighbors
            two_hop_neighbors = set()
            neighbors_u = list(graph.neighbors(u))
            
            # Sample b neighbors nếu có quá nhiều
            if len(neighbors_u) > self.b:
                neighbors_u = random.sample(neighbors_u, self.b)
            
            two_hop_neighbors.update(neighbors_u)
            for neighbor in neighbors_u:
                two_hop_neighbors.update(graph.neighbors(neighbor))
            
            # Tính similarity với các 2-hop neighbors
            for v in two_hop_neighbors:
                if u != v:
                    # Tính similarity dựa trên MinHash
                    mh_u = minhash_signatures[u]
                    mh_v = minhash_signatures[v]
                    mh_similarity = sum(1 for x, y in zip(mh_u, mh_v) if x == y) / self.h
                    
                    similarities.append((v, mh_similarity))
            
            # Chọn top k candidates
            similarities.sort(key=lambda x: x[1], reverse=True)
            for v, sim in similarities[:self.k]:
                candidates.append((u, v, sim))
        
        return candidates
    
    def _compute_saving(self, graph: nx.Graph, u: int, v: int) -> float:
        """Tính saving khi merge hai nodes u và v"""
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        
        # Cost trước khi merge
        cost_u = len(neighbors_u)
        cost_v = len(neighbors_v)
        cost_before = cost_u + cost_v
        
        if cost_before == 0:
            return 0.0
        
        # Cost sau khi merge 
        merged_neighbors = neighbors_u | neighbors_v
        if u in merged_neighbors:
            merged_neighbors.remove(u)
        if v in merged_neighbors:
            merged_neighbors.remove(v)
        cost_after = len(merged_neighbors)
        
        # Saving = (cost_before - cost_after) / cost_before
        saving = (cost_before - cost_after) / cost_before
        return saving
    
    def _merge_threshold(self, t: int) -> float:
        """Tính merge threshold ω(t) cho iteration t"""
        if t >= self.T:
            return 0.005
        else:
            r = (0.01) ** (1 / (self.T - 1))
            return 0.5 * (r ** (t - 1))
    
    def _greedy_merge(self, graph: nx.Graph, candidates: List[Tuple[int, int, float]]) -> Dict[int, SuperNode]:
        """
        Phase 2: Greedy Merge
        Merge các nodes thành super-nodes dựa trên saving
        """
        print("Starting greedy merge phase...")
        
        # Khởi tạo: mỗi node là một super-node
        super_nodes = {}
        node_to_super = {}  # mapping từ original node to super-node id
        
        for node in graph.nodes():
            super_id = node  # Sử dụng original node id làm super-node id
            super_nodes[super_id] = SuperNode(super_id, {node})
            node_to_super[node] = super_id
        
        # Sort candidates theo saving giảm dần
        candidates_with_saving = []
        for u, v, sim in candidates:
            saving = self._compute_saving(graph, u, v)
            candidates_with_saving.append((u, v, saving, sim))
        
        candidates_with_saving.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy merge qua T iterations
        for t in range(1, self.T + 1):
            threshold = self._merge_threshold(t)
            print(f"Iteration {t}/{self.T}, threshold = {threshold:.6f}")
            
            merged_pairs = 0
            for u, v, saving, sim in candidates_with_saving:
                # Kiểm tra nếu hai nodes chưa được merge
                super_u = node_to_super.get(u)
                super_v = node_to_super.get(v)
                
                if super_u is None or super_v is None or super_u == super_v:
                    continue
                
                # Re-compute saving với current state
                current_saving = self._compute_saving(graph, u, v)
                
                if current_saving >= threshold:
                    # Merge super_v vào super_u
                    super_nodes[super_u].nodes.update(super_nodes[super_v].nodes)
                    super_nodes[super_u].size = len(super_nodes[super_u].nodes)
                    
                    # Update mapping cho tất cả nodes trong super_v
                    for node in super_nodes[super_v].nodes:
                        node_to_super[node] = super_u
                    
                    # Xóa super_v
                    del super_nodes[super_v]
                    merged_pairs += 1
            
            print(f"  Merged {merged_pairs} pairs, remaining super-nodes: {len(super_nodes)}")
            
            if merged_pairs == 0:
                break
        
        return super_nodes
    
    def summarize(self, graph: nx.Graph) -> Dict[int, SuperNode]:
        """
        Thực hiện graph summarization
        
        Args:
            graph: Input graph
            
        Returns:
            Dictionary của super-nodes
        """
        print(f"Starting MAGS summarization on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        start_time = time.time()
        
        # Phase 1: Generate candidate pairs
        candidates = self._generate_candidate_pairs(graph)
        print(f"Generated {len(candidates)} candidate pairs")
        
        # Phase 2: Greedy merge
        super_nodes = self._greedy_merge(graph, candidates)
        
        end_time = time.time()
        print(f"Summarization completed in {end_time - start_time:.2f} seconds")
        print(f"Original graph: {graph.number_of_nodes()} nodes -> Summary: {len(super_nodes)} super-nodes")
        print(f"Compression ratio: {(1 - len(super_nodes) / graph.number_of_nodes()) * 100:.2f}%")
        
        return super_nodes
    
    def get_summary_stats(self, original_graph: nx.Graph, super_nodes: Dict[int, SuperNode]) -> Dict:
        """Tính toán thống kê của summary"""
        original_nodes = original_graph.number_of_nodes()
        original_edges = original_graph.number_of_edges()
        summary_nodes = len(super_nodes)
        
        # Tính số super-edges (đơn giản hóa)
        super_edges = 0
        for super_node in super_nodes.values():
            for other_super in super_nodes.values():
                if super_node.id != other_super.id:
                    # Kiểm tra nếu có edge giữa hai super-nodes
                    has_edge = False
                    for u in super_node.nodes:
                        for v in other_super.nodes:
                            if original_graph.has_edge(u, v):
                                has_edge = True
                                break
                        if has_edge:
                            break
                    if has_edge:
                        super_edges += 1
        
        super_edges //= 2  # Undirected graph
        
        compression_ratio = (1 - summary_nodes / original_nodes) * 100
        
        return {
            'original_nodes': original_nodes,
            'original_edges': original_edges,
            'summary_nodes': summary_nodes,
            'summary_edges': super_edges,
            'compression_ratio': compression_ratio,
            'avg_supernode_size': sum(sn.size for sn in super_nodes.values()) / len(super_nodes)
        }

def experiment_with_internet_graphs():
    """Thực nghiệm với các đồ thị internet"""
    from download_internet_graphs import InternetGraphDownloader
    
    print("=== THỰC NGHIỆM MAGS TRÊN CÁC ĐỒ THỊ INTERNET ===\n")
    
    # Tải datasets
    downloader = InternetGraphDownloader()
    datasets = downloader.download_and_prepare_datasets()
    
    if not datasets:
        print("Không thể tải datasets!")
        return
    
    # Khởi tạo MAGS
    mags = MAGS(k=5, T=20, b=5, h=10)  # Giảm T để test nhanh hơn
    
    results = {}
    
    for name, graph in datasets.items():
        print(f"\n{'='*50}")
        print(f"Thực nghiệm với dataset: {name.upper()}")
        print(f"{'='*50}")
        
        # Chỉ test với graphs nhỏ hơn để demo
        if graph.number_of_nodes() > 10000:
            print(f"Graph quá lớn ({graph.number_of_nodes()} nodes), tạo subgraph để test...")
            # Lấy subgraph nhỏ hơn
            nodes = list(graph.nodes())[:5000]
            graph = graph.subgraph(nodes).copy()
        
        try:
            # Chạy MAGS summarization
            super_nodes = mags.summarize(graph)
            
            # Tính thống kê
            stats = mags.get_summary_stats(graph, super_nodes)
            results[name] = stats
            
            print(f"\nKết quả cho {name}:")
            print(f"  Original: {stats['original_nodes']} nodes, {stats['original_edges']} edges")
            print(f"  Summary: {stats['summary_nodes']} super-nodes, {stats['summary_edges']} super-edges")
            print(f"  Compression ratio: {stats['compression_ratio']:.2f}%")
            print(f"  Average super-node size: {stats['avg_supernode_size']:.2f}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý {name}: {e}")
            continue
    
    # In tổng kết
    print(f"\n{'='*60}")
    print("TỔNG KẾT KẾT QUẢ")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Orig.Nodes':<12} {'Sum.Nodes':<12} {'Compression':<12}")
    print("-" * 60)
    
    for name, stats in results.items():
        print(f"{name:<15} {stats['original_nodes']:<12} {stats['summary_nodes']:<12} {stats['compression_ratio']:<12.2f}%")

if __name__ == "__main__":
    experiment_with_internet_graphs() 