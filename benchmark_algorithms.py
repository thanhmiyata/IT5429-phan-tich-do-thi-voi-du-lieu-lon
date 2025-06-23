#!/usr/bin/env python3
"""
Benchmark suite cho Graph Summarization Algorithms
Bao gồm: MAGS, MAGS-DM, Greedy Baseline, LDME
Đánh giá: Compression ratio, Runtime, Query efficiency
"""

import numpy as np
import networkx as nx
from collections import defaultdict, deque
import random
import math
import time
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
from mags_implementation import MAGS, SuperNode

class MAGSDivideAndMerge:
    """
    Implementation của MAGS-DM (Divide-and-Merge) algorithm
    Dựa trên paper SIGMOD 2024
    """
    
    def __init__(self, k: int = 5, T: int = 50, b: int = 5, h: int = 10, partition_size: int = 1000):
        self.k = k
        self.T = T
        self.b = b
        self.h = h
        self.partition_size = partition_size
        self.mags = MAGS(k, T, b, h)
    
    def _partition_graph(self, graph: nx.Graph) -> List[nx.Graph]:
        """Chia graph thành các partitions nhỏ hơn"""
        nodes = list(graph.nodes())
        n_partitions = max(1, len(nodes) // self.partition_size)
        
        if n_partitions == 1:
            return [graph]
        
        # Random partitioning (có thể cải thiện bằng community detection)
        random.shuffle(nodes)
        partitions = []
        
        for i in range(n_partitions):
            start_idx = i * self.partition_size
            end_idx = min((i + 1) * self.partition_size, len(nodes))
            partition_nodes = nodes[start_idx:end_idx]
            
            # Tạo subgraph
            subgraph = graph.subgraph(partition_nodes).copy()
            partitions.append(subgraph)
        
        return partitions
    
    def _merge_summaries(self, summaries: List[Dict[int, SuperNode]]) -> Dict[int, SuperNode]:
        """Merge các summaries từ different partitions"""
        merged_summary = {}
        super_node_id = 0
        
        for summary in summaries:
            for _, super_node in summary.items():
                merged_summary[super_node_id] = SuperNode(super_node_id, super_node.nodes.copy())
                super_node_id += 1
        
        return merged_summary
    
    def summarize(self, graph: nx.Graph) -> Dict[int, SuperNode]:
        """
        MAGS-DM main algorithm
        """
        print(f"Starting MAGS-DM on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Phase 1: Divide
        partitions = self._partition_graph(graph)
        print(f"Divided into {len(partitions)} partitions")
        
        # Phase 2: Summarize each partition
        summaries = []
        for i, partition in enumerate(partitions):
            print(f"Processing partition {i+1}/{len(partitions)} ({partition.number_of_nodes()} nodes)")
            summary = self.mags.summarize(partition)
            summaries.append(summary)
        
        # Phase 3: Merge summaries
        final_summary = self._merge_summaries(summaries)
        
        print(f"MAGS-DM completed: {len(final_summary)} super-nodes")
        return final_summary

class GreedyBaseline:
    """
    Greedy baseline algorithm cho graph summarization
    """
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
    
    def _compute_saving(self, graph: nx.Graph, u: int, v: int) -> float:
        """Tính saving khi merge hai nodes"""
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        
        cost_before = len(neighbors_u) + len(neighbors_v)
        if cost_before == 0:
            return 0.0
        
        merged_neighbors = neighbors_u | neighbors_v
        if u in merged_neighbors:
            merged_neighbors.remove(u)
        if v in merged_neighbors:
            merged_neighbors.remove(v)
        
        cost_after = len(merged_neighbors)
        return (cost_before - cost_after) / cost_before
    
    def summarize(self, graph: nx.Graph) -> Dict[int, SuperNode]:
        """Simple greedy merging"""
        print(f"Starting Greedy baseline on graph with {graph.number_of_nodes()} nodes")
        
        # Initialize: mỗi node là một super-node
        super_nodes = {}
        node_to_super = {}
        
        for node in graph.nodes():
            super_nodes[node] = SuperNode(node, {node})
            node_to_super[node] = node
        
        # Greedy merge
        for iteration in range(self.max_iterations):
            best_saving = 0
            best_pair = None
            
            # Tìm best pair để merge
            for u in list(super_nodes.keys()):
                for v in list(super_nodes.keys()):
                    if u >= v:
                        continue
                    
                    saving = self._compute_saving(graph, u, v)
                    if saving > best_saving:
                        best_saving = saving
                        best_pair = (u, v)
            
            if best_pair is None or best_saving <= 0:
                break
            
            # Merge best pair
            u, v = best_pair
            if u in super_nodes and v in super_nodes:
                super_nodes[u].nodes.update(super_nodes[v].nodes)
                del super_nodes[v]
                
                # Update mapping
                for node in super_nodes[u].nodes:
                    node_to_super[node] = u
        
        print(f"Greedy completed: {len(super_nodes)} super-nodes after {iteration+1} iterations")
        return super_nodes

class LDMEApproximation:
    """
    Approximation của LDME (Lossless Directed Multigraph Encoding)
    Simplified version for comparison
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def _structural_similarity(self, graph: nx.Graph, u: int, v: int) -> float:
        """Tính structural similarity giữa hai nodes"""
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        
        if not neighbors_u and not neighbors_v:
            return 1.0
        if not neighbors_u or not neighbors_v:
            return 0.0
        
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        
        return intersection / union if union > 0 else 0.0
    
    def summarize(self, graph: nx.Graph) -> Dict[int, SuperNode]:
        """LDME-style summarization"""
        print(f"Starting LDME approximation on graph with {graph.number_of_nodes()} nodes")
        
        super_nodes = {}
        node_to_super = {}
        
        # Initialize
        for node in graph.nodes():
            super_nodes[node] = SuperNode(node, {node})
            node_to_super[node] = node
        
        # Merge based on structural similarity
        nodes = list(graph.nodes())
        for i, u in enumerate(nodes):
            if u not in super_nodes:
                continue
                
            for j, v in enumerate(nodes[i+1:], i+1):
                if v not in super_nodes:
                    continue
                
                similarity = self._structural_similarity(graph, u, v)
                if similarity >= self.threshold:
                    # Merge v into u
                    if u in super_nodes and v in super_nodes:
                        super_nodes[u].nodes.update(super_nodes[v].nodes)
                        del super_nodes[v]
                        
                        for node in super_nodes[u].nodes:
                            node_to_super[node] = u
        
        print(f"LDME completed: {len(super_nodes)} super-nodes")
        return super_nodes

class QueryEfficiencyEvaluator:
    """
    Đánh giá hiệu quả truy vấn trên original vs summary graph
    """
    
    def __init__(self):
        self.query_types = [
            'shortest_path',
            'degree_centrality', 
            'clustering_coefficient',
            'connected_components',
            'node_neighbors'
        ]
    
    def _generate_random_queries(self, graph: nx.Graph, n_queries: int = 100) -> List[Dict]:
        """Tạo random queries để test"""
        nodes = list(graph.nodes())
        queries = []
        
        for _ in range(n_queries):
            query_type = random.choice(self.query_types)
            
            if query_type == 'shortest_path':
                u, v = random.sample(nodes, 2)
                queries.append({'type': 'shortest_path', 'source': u, 'target': v})
            
            elif query_type == 'degree_centrality':
                node = random.choice(nodes)
                queries.append({'type': 'degree_centrality', 'node': node})
            
            elif query_type == 'clustering_coefficient':
                node = random.choice(nodes)
                queries.append({'type': 'clustering_coefficient', 'node': node})
            
            elif query_type == 'connected_components':
                queries.append({'type': 'connected_components'})
            
            elif query_type == 'node_neighbors':
                node = random.choice(nodes)
                queries.append({'type': 'node_neighbors', 'node': node})
        
        return queries
    
    def _execute_query(self, graph: nx.Graph, query: Dict) -> Tuple[float, any]:
        """Execute một query và đo thời gian"""
        start_time = time.time()
        result = None
        
        try:
            if query['type'] == 'shortest_path':
                if nx.has_path(graph, query['source'], query['target']):
                    result = nx.shortest_path_length(graph, query['source'], query['target'])
                else:
                    result = float('inf')
            
            elif query['type'] == 'degree_centrality':
                if query['node'] in graph:
                    result = graph.degree(query['node'])
                else:
                    result = 0
            
            elif query['type'] == 'clustering_coefficient':
                if query['node'] in graph:
                    result = nx.clustering(graph, query['node'])
                else:
                    result = 0
            
            elif query['type'] == 'connected_components':
                result = nx.number_connected_components(graph)
            
            elif query['type'] == 'node_neighbors':
                if query['node'] in graph:
                    result = len(list(graph.neighbors(query['node'])))
                else:
                    result = 0
        
        except Exception as e:
            result = None
        
        execution_time = time.time() - start_time
        return execution_time, result
    
    def evaluate_query_efficiency(self, original_graph: nx.Graph, 
                                 summary_nodes: Dict[int, SuperNode],
                                 n_queries: int = 50) -> Dict:
        """Đánh giá query efficiency"""
        
        # Tạo summary graph
        summary_graph = nx.Graph()
        for super_id, super_node in summary_nodes.items():
            summary_graph.add_node(super_id)
        
        # Add edges between super-nodes (simplified)
        for super_id1, super_node1 in summary_nodes.items():
            for super_id2, super_node2 in summary_nodes.items():
                if super_id1 >= super_id2:
                    continue
                
                # Check if there's any edge between the two super-nodes
                has_edge = False
                for node1 in super_node1.nodes:
                    for node2 in super_node2.nodes:
                        if original_graph.has_edge(node1, node2):
                            has_edge = True
                            break
                    if has_edge:
                        break
                
                if has_edge:
                    summary_graph.add_edge(super_id1, super_id2)
        
        # Generate queries
        queries = self._generate_random_queries(original_graph, n_queries)
        
        # Execute queries on both graphs
        original_times = []
        summary_times = []
        
        for query in queries:
            # Original graph
            orig_time, orig_result = self._execute_query(original_graph, query)
            original_times.append(orig_time)
            
            # Summary graph (adapt query)
            adapted_query = self._adapt_query_for_summary(query, summary_nodes)
            if adapted_query:
                summ_time, summ_result = self._execute_query(summary_graph, adapted_query)
                summary_times.append(summ_time)
            else:
                summary_times.append(0)  # Query not applicable
        
        return {
            'original_avg_time': np.mean(original_times),
            'summary_avg_time': np.mean(summary_times),
            'speedup': np.mean(original_times) / np.mean(summary_times) if np.mean(summary_times) > 0 else float('inf'),
            'original_graph_size': original_graph.number_of_nodes(),
            'summary_graph_size': summary_graph.number_of_nodes()
        }
    
    def _adapt_query_for_summary(self, query: Dict, summary_nodes: Dict[int, SuperNode]) -> Dict:
        """Adapt query để chạy trên summary graph"""
        # Tạo mapping từ original node to super-node
        node_to_super = {}
        for super_id, super_node in summary_nodes.items():
            for node in super_node.nodes:
                node_to_super[node] = super_id
        
        if query['type'] == 'shortest_path':
            source_super = node_to_super.get(query['source'])
            target_super = node_to_super.get(query['target'])
            if source_super is not None and target_super is not None:
                return {'type': 'shortest_path', 'source': source_super, 'target': target_super}
        
        elif query['type'] in ['degree_centrality', 'clustering_coefficient', 'node_neighbors']:
            node_super = node_to_super.get(query['node'])
            if node_super is not None:
                return {'type': query['type'], 'node': node_super}
        
        elif query['type'] == 'connected_components':
            return query
        
        return None

class ComprehensiveBenchmark:
    """
    Comprehensive benchmark suite cho tất cả algorithms
    """
    
    def __init__(self):
        self.algorithms = {
            'MAGS': MAGS(k=5, T=20, b=3, h=30),
            'MAGS-DM': MAGSDivideAndMerge(k=5, T=20, b=3, h=30, partition_size=1000),
            'Greedy': GreedyBaseline(max_iterations=50),
            'LDME': LDMEApproximation(threshold=0.2)
        }
        
        self.query_evaluator = QueryEfficiencyEvaluator()
    
    def run_single_benchmark(self, graph: nx.Graph, dataset_name: str, 
                           max_nodes: int = 2000) -> Dict:
        """Chạy benchmark trên một dataset"""
        
        # Limit graph size for reasonable runtime
        if graph.number_of_nodes() > max_nodes:
            print(f"Graph quá lớn ({graph.number_of_nodes()} nodes), sampling {max_nodes} nodes")
            nodes = random.sample(list(graph.nodes()), max_nodes)
            graph = graph.subgraph(nodes).copy()
        
        results = {
            'dataset': dataset_name,
            'original_nodes': graph.number_of_nodes(),
            'original_edges': graph.number_of_edges()
        }
        
        print(f"\n=== BENCHMARKING {dataset_name.upper()} ===")
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        for alg_name, algorithm in self.algorithms.items():
            print(f"\n--- Running {alg_name} ---")
            
            try:
                # Measure runtime
                start_time = time.time()
                summary = algorithm.summarize(graph)
                runtime = time.time() - start_time
                
                # Compute compression ratio
                compression_ratio = (1 - len(summary) / graph.number_of_nodes()) * 100
                
                # Evaluate query efficiency
                query_stats = self.query_evaluator.evaluate_query_efficiency(
                    graph, summary, n_queries=30
                )
                
                results[f'{alg_name}_runtime'] = runtime
                results[f'{alg_name}_summary_nodes'] = len(summary)
                results[f'{alg_name}_compression_ratio'] = compression_ratio
                results[f'{alg_name}_query_speedup'] = query_stats['speedup']
                results[f'{alg_name}_original_query_time'] = query_stats['original_avg_time']
                results[f'{alg_name}_summary_query_time'] = query_stats['summary_avg_time']
                
                print(f"  Summary: {len(summary)} super-nodes")
                print(f"  Compression: {compression_ratio:.2f}%")
                print(f"  Runtime: {runtime:.3f}s")
                print(f"  Query speedup: {query_stats['speedup']:.2f}x")
                
            except Exception as e:
                print(f"  Error running {alg_name}: {e}")
                results[f'{alg_name}_runtime'] = float('inf')
                results[f'{alg_name}_summary_nodes'] = graph.number_of_nodes()
                results[f'{alg_name}_compression_ratio'] = 0.0
                results[f'{alg_name}_query_speedup'] = 1.0
                results[f'{alg_name}_original_query_time'] = 0.0
                results[f'{alg_name}_summary_query_time'] = 0.0
        
        return results
    
    def run_comprehensive_benchmark(self, graphs: Dict[str, nx.Graph]) -> pd.DataFrame:
        """Chạy benchmark trên tất cả datasets"""
        all_results = []
        
        for dataset_name, graph in graphs.items():
            result = self.run_single_benchmark(graph, dataset_name)
            all_results.append(result)
        
        return pd.DataFrame(all_results)

def main():
    """Test benchmark system"""
    from extended_datasets import ExtendedInternetGraphDownloader
    
    # Download datasets (chỉ test với một vài datasets nhỏ)
    downloader = ExtendedInternetGraphDownloader()
    
    # Test với graph nhỏ
    test_graphs = {
        'karate': nx.karate_club_graph(),
        'les_mis': nx.les_miserables_graph()
    }
    
    # Run benchmark
    benchmark = ComprehensiveBenchmark()
    results_df = benchmark.run_comprehensive_benchmark(test_graphs)
    
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('benchmark_results.csv', index=False)
    print(f"\nĐã lưu kết quả vào benchmark_results.csv")

if __name__ == "__main__":
    main() 