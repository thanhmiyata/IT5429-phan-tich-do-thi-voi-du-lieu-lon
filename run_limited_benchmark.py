#!/usr/bin/env python3
"""
Limited benchmark với 10 internet datasets (sử dụng subsets nhỏ hơn)
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extended_datasets import ExtendedInternetGraphDownloader
from benchmark_algorithms import ComprehensiveBenchmark
import numpy as np
import random

def create_limited_visualization(results_df: pd.DataFrame, output_dir: str = "limited_benchmark_results"):
    """Tạo visualization cho limited benchmark"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    algorithms = ['MAGS', 'MAGS-DM', 'Greedy', 'LDME']
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Compression Ratio Comparison
    compression_data = []
    for alg in algorithms:
        col = f'{alg}_compression_ratio'
        if col in results_df.columns:
            for idx, row in results_df.iterrows():
                if row[col] != float('inf') and row[col] > 0:
                    compression_data.append({
                        'Algorithm': alg,
                        'Dataset': row['dataset'],
                        'Compression_Ratio': row[col]
                    })
    
    if compression_data:
        compression_df = pd.DataFrame(compression_data)
        sns.barplot(data=compression_df, x='Dataset', y='Compression_Ratio', hue='Algorithm', ax=axes[0,0])
        axes[0,0].set_title('Compression Ratio Comparison (%)', fontsize=14)
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylabel('Compression Ratio (%)')
    
    # 2. Runtime Comparison
    runtime_data = []
    for alg in algorithms:
        col = f'{alg}_runtime'
        if col in results_df.columns:
            for idx, row in results_df.iterrows():
                if row[col] != float('inf') and row[col] > 0:
                    runtime_data.append({
                        'Algorithm': alg,
                        'Dataset': row['dataset'],
                        'Runtime': row[col]
                    })
    
    if runtime_data:
        runtime_df = pd.DataFrame(runtime_data)
        sns.barplot(data=runtime_df, x='Dataset', y='Runtime', hue='Algorithm', ax=axes[0,1])
        axes[0,1].set_title('Runtime Comparison (seconds)', fontsize=14)
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylabel('Runtime (s)')
        axes[0,1].set_yscale('log')
    
    # 3. Query Speedup Comparison
    speedup_data = []
    for alg in algorithms:
        col = f'{alg}_query_speedup'
        if col in results_df.columns:
            for idx, row in results_df.iterrows():
                if row[col] != float('inf') and row[col] > 0:
                    speedup_data.append({
                        'Algorithm': alg,
                        'Dataset': row['dataset'],
                        'Query_Speedup': row[col]
                    })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        sns.barplot(data=speedup_df, x='Dataset', y='Query_Speedup', hue='Algorithm', ax=axes[1,0])
        axes[1,0].set_title('Query Speedup Comparison', fontsize=14)  
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_ylabel('Speedup (x)')
    
    # 4. Summary Nodes vs Original Nodes
    for alg in algorithms:
        col = f'{alg}_summary_nodes'
        if col in results_df.columns:
            valid_data = results_df[results_df[col] != float('inf')]
            if not valid_data.empty:
                axes[1,1].scatter(valid_data['original_nodes'], valid_data[col], 
                               label=alg, alpha=0.7, s=80)
    
    max_nodes = results_df['original_nodes'].max()
    axes[1,1].plot([0, max_nodes], [0, max_nodes], 'k--', alpha=0.5, label='No compression')
    axes[1,1].set_xlabel('Original Nodes')
    axes[1,1].set_ylabel('Summary Nodes') 
    axes[1,1].set_title('Compression Effectiveness', fontsize=14)
    axes[1,1].legend()
    axes[1,1].set_xscale('log')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/limited_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results_df.to_csv(f'{output_dir}/detailed_results.csv', index=False)
    print(f"📊 Visualization saved to: {output_dir}/limited_benchmark_comparison.png")

def main():
    """Chạy limited benchmark với 10 internet datasets"""
    
    print("🚀 STARTING LIMITED COMPREHENSIVE BENCHMARK")
    print("📊 10 Internet Graph Datasets with Reasonable Subsets")
    print("=" * 80)
    
    # Step 1: Load datasets
    print("\n📥 STEP 1: LOADING DATASETS")
    downloader = ExtendedInternetGraphDownloader()
    graphs = downloader.download_all_datasets()
    
    if not graphs:
        print("❌ Không thể tải datasets.")
        return
    
    print(f"✅ Loaded {len(graphs)} datasets")
    
    # Step 2: Create reasonable subsets for benchmarking
    print("\n🔧 STEP 2: CREATING REASONABLE SUBSETS")
    limited_graphs = {}
    max_nodes_per_dataset = 1500  # Reasonable size for comprehensive testing
    
    for name, graph in graphs.items():
        original_size = graph.number_of_nodes()
        
        if original_size <= max_nodes_per_dataset:
            limited_graphs[name] = graph
            print(f"  {name}: Using full graph ({original_size} nodes)")
        else:
            # Sample a connected subgraph
            # Get largest connected component first
            largest_cc = max(nx.connected_components(graph), key=len)
            cc_graph = graph.subgraph(largest_cc).copy()
            
            if len(largest_cc) <= max_nodes_per_dataset:
                limited_graphs[name] = cc_graph
                print(f"  {name}: Using largest CC ({len(largest_cc)} nodes from {original_size})")
            else:
                # Sample from largest CC
                sampled_nodes = random.sample(list(largest_cc), max_nodes_per_dataset)
                sampled_graph = graph.subgraph(sampled_nodes).copy()
                
                # Ensure connectivity by taking largest CC of sample
                if nx.is_connected(sampled_graph):
                    limited_graphs[name] = sampled_graph
                else:
                    sample_cc = max(nx.connected_components(sampled_graph), key=len)
                    limited_graphs[name] = sampled_graph.subgraph(sample_cc).copy()
                
                final_size = limited_graphs[name].number_of_nodes()
                print(f"  {name}: Sampled subset ({final_size} nodes from {original_size})")
    
    # Step 3: Show dataset statistics
    print(f"\n📊 STEP 3: LIMITED DATASET STATISTICS")
    stats_data = []
    for name, graph in limited_graphs.items():
        stats_data.append({
            'Dataset': name,
            'Nodes': graph.number_of_nodes(),
            'Edges': graph.number_of_edges(),
            'Avg_Degree': 2 * graph.number_of_edges() / graph.number_of_nodes(),
            'Density': nx.density(graph)
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Step 4: Run comprehensive benchmark
    print(f"\n⚡ STEP 4: RUNNING COMPREHENSIVE BENCHMARK")
    print("Algorithms: MAGS, MAGS-DM, Greedy Baseline, LDME Approximation")
    print("Metrics: Compression Ratio, Runtime, Query Efficiency")
    
    benchmark = ComprehensiveBenchmark()
    
    start_time = time.time()
    results_df = benchmark.run_comprehensive_benchmark(limited_graphs)
    total_time = time.time() - start_time
    
    print(f"\n✅ Benchmark completed in {total_time:.2f} seconds")
    
    # Step 5: Create analysis
    print(f"\n📈 STEP 5: CREATING ANALYSIS")
    create_limited_visualization(results_df)
    
    # Step 6: Summary results
    print(f"\n🏆 STEP 6: COMPREHENSIVE RESULTS SUMMARY")
    
    algorithms = ['MAGS', 'MAGS-DM', 'Greedy', 'LDME']
    
    # Compression ratios
    print(f"\nCOMPRESSION RATIOS:")
    print(f"{'Dataset':<20} {'Nodes':<8} {'Edges':<8}", end="")
    for alg in algorithms:
        print(f" {alg:<12}", end="")
    print()
    print("-" * (20 + 8 + 8 + 12 * len(algorithms)))
    
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:<20} {row['original_nodes']:<8} {row['original_edges']:<8}", end="")
        for alg in algorithms:
            col = f'{alg}_compression_ratio'
            if col in row and row[col] != float('inf'):
                print(f" {row[col]:<11.1f}%", end="")
            else:
                print(f" {'N/A':<12}", end="")
        print()
    
    # Runtime comparison
    print(f"\nRUNTIME COMPARISON (seconds):")
    print(f"{'Dataset':<20}", end="")
    for alg in algorithms:
        print(f" {alg:<12}", end="")
    print()
    print("-" * (20 + 12 * len(algorithms)))
    
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:<20}", end="")
        for alg in algorithms:
            col = f'{alg}_runtime'
            if col in row and row[col] != float('inf'):
                print(f" {row[col]:<11.3f}s", end="")
            else:
                print(f" {'N/A':<12}", end="")
        print()
    
    # Query speedup
    print(f"\nQUERY SPEEDUP:")
    print(f"{'Dataset':<20}", end="")
    for alg in algorithms:
        print(f" {alg:<12}", end="")
    print()
    print("-" * (20 + 12 * len(algorithms)))
    
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:<20}", end="")
        for alg in algorithms:
            col = f'{alg}_query_speedup'
            if col in row and row[col] != float('inf'):
                print(f" {row[col]:<11.2f}x", end="")
            else:
                print(f" {'N/A':<12}", end="")
        print()
    
    # Step 7: Algorithm ranking
    print(f"\n🥇 STEP 7: ALGORITHM RANKING")
    
    # Calculate average metrics
    avg_metrics = {}
    for alg in algorithms:
        compression_col = f'{alg}_compression_ratio'
        runtime_col = f'{alg}_runtime'
        speedup_col = f'{alg}_query_speedup'
        
        valid_data = results_df[
            (results_df[compression_col] != float('inf')) & 
            (results_df[runtime_col] != float('inf')) &
            (results_df[speedup_col] != float('inf'))
        ]
        
        if not valid_data.empty:
            avg_metrics[alg] = {
                'avg_compression': valid_data[compression_col].mean(),
                'avg_runtime': valid_data[runtime_col].mean(),
                'avg_speedup': valid_data[speedup_col].mean(),
                'success_rate': len(valid_data) / len(results_df)
            }
    
    print(f"\nAVERAGE PERFORMANCE:")
    print(f"{'Algorithm':<12} {'Compression':<12} {'Runtime':<12} {'Query Speedup':<14} {'Success Rate':<12}")
    print("-" * 70)
    
    for alg, metrics in avg_metrics.items():
        print(f"{alg:<12} {metrics['avg_compression']:<11.1f}% {metrics['avg_runtime']:<11.3f}s "
              f"{metrics['avg_speedup']:<13.2f}x {metrics['success_rate']:<11.1%}")
    
    print(f"\n🎯 BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved in: limited_benchmark_results/")
    print(f"📊 Detailed CSV: limited_benchmark_results/detailed_results.csv")
    print(f"📈 Visualization: limited_benchmark_results/limited_benchmark_comparison.png")

if __name__ == "__main__":
    import networkx as nx
    main() 