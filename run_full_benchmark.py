#!/usr/bin/env python3
"""
Script chính để chạy comprehensive benchmark với 10 internet graph datasets
So sánh MAGS, MAGS-DM, Greedy, LDME về compression, runtime, query efficiency
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extended_datasets import ExtendedInternetGraphDownloader
from benchmark_algorithms import ComprehensiveBenchmark
import numpy as np

def create_visualization(results_df: pd.DataFrame, output_dir: str = "benchmark_results"):
    """Tạo các biểu đồ visualization cho kết quả benchmark"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    algorithms = ['MAGS', 'MAGS-DM', 'Greedy', 'LDME']
    
    # 1. Compression Ratio Comparison
    plt.figure(figsize=(15, 8))
    
    compression_data = []
    for alg in algorithms:
        col = f'{alg}_compression_ratio'
        if col in results_df.columns:
            for idx, row in results_df.iterrows():
                compression_data.append({
                    'Algorithm': alg,
                    'Dataset': row['dataset'],
                    'Compression_Ratio': row[col]
                })
    
    compression_df = pd.DataFrame(compression_data)
    
    plt.subplot(2, 2, 1)
    sns.barplot(data=compression_df, x='Dataset', y='Compression_Ratio', hue='Algorithm')
    plt.title('Compression Ratio Comparison (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Compression Ratio (%)')
    
    # 2. Runtime Comparison
    plt.subplot(2, 2, 2)
    runtime_data = []
    for alg in algorithms:
        col = f'{alg}_runtime'
        if col in results_df.columns:
            for idx, row in results_df.iterrows():
                if row[col] != float('inf'):
                    runtime_data.append({
                        'Algorithm': alg,
                        'Dataset': row['dataset'],
                        'Runtime': row[col]
                    })
    
    runtime_df = pd.DataFrame(runtime_data)
    if not runtime_df.empty:
        sns.barplot(data=runtime_df, x='Dataset', y='Runtime', hue='Algorithm')
        plt.title('Runtime Comparison (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Runtime (s)')
        plt.yscale('log')
    
    # 3. Query Speedup Comparison
    plt.subplot(2, 2, 3)
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
    
    speedup_df = pd.DataFrame(speedup_data)
    if not speedup_df.empty:
        sns.barplot(data=speedup_df, x='Dataset', y='Query_Speedup', hue='Algorithm')
        plt.title('Query Speedup Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Speedup (x)')
    
    # 4. Summary Nodes vs Original Nodes
    plt.subplot(2, 2, 4)
    for alg in algorithms:
        col = f'{alg}_summary_nodes'
        if col in results_df.columns:
            plt.scatter(results_df['original_nodes'], results_df[col], 
                       label=alg, alpha=0.7, s=60)
    
    plt.plot([0, results_df['original_nodes'].max()], 
             [0, results_df['original_nodes'].max()], 
             'k--', alpha=0.5, label='No compression')
    plt.xlabel('Original Nodes')
    plt.ylabel('Summary Nodes')
    plt.title('Compression Effectiveness')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed Performance Table
    create_performance_table(results_df, output_dir)
    
    print(f"Đã tạo visualization trong thư mục {output_dir}/")

def create_performance_table(results_df: pd.DataFrame, output_dir: str):
    """Tạo bảng performance chi tiết"""
    algorithms = ['MAGS', 'MAGS-DM', 'Greedy', 'LDME']
    
    # Tạo summary table
    summary_data = []
    
    for alg in algorithms:
        runtime_col = f'{alg}_runtime'
        compression_col = f'{alg}_compression_ratio'
        speedup_col = f'{alg}_query_speedup'
        
        if all(col in results_df.columns for col in [runtime_col, compression_col, speedup_col]):
            # Filter out infinite values
            valid_data = results_df[
                (results_df[runtime_col] != float('inf')) & 
                (results_df[compression_col] > 0) &
                (results_df[speedup_col] != float('inf'))
            ]
            
            if not valid_data.empty:
                summary_data.append({
                    'Algorithm': alg,
                    'Avg_Compression_Ratio': valid_data[compression_col].mean(),
                    'Std_Compression_Ratio': valid_data[compression_col].std(),
                    'Avg_Runtime': valid_data[runtime_col].mean(),
                    'Std_Runtime': valid_data[runtime_col].std(),
                    'Avg_Query_Speedup': valid_data[speedup_col].mean(),
                    'Std_Query_Speedup': valid_data[speedup_col].std(),
                    'Success_Rate': len(valid_data) / len(results_df)
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save detailed results
    results_df.to_csv(f'{output_dir}/detailed_results.csv', index=False)
    summary_df.to_csv(f'{output_dir}/summary_results.csv', index=False)
    
    # Create formatted table
    if not summary_df.empty:
        print(f"\n{'='*100}")
        print("TỔNG KẾT PERFORMANCE TRUNG BÌNH")
        print(f"{'='*100}")
        
        formatted_summary = summary_df.copy()
        for col in ['Avg_Compression_Ratio', 'Avg_Runtime', 'Avg_Query_Speedup']:
            formatted_summary[col] = formatted_summary[col].round(3)
        
        print(formatted_summary.to_string(index=False))

def main():
    """Chạy full benchmark với 10 internet graph datasets"""
    
    print("🚀 STARTING COMPREHENSIVE GRAPH SUMMARIZATION BENCHMARK")
    print("=" * 80)
    
    # Step 1: Download all datasets
    print("\n📥 STEP 1: DOWNLOADING 10 INTERNET GRAPH DATASETS")
    downloader = ExtendedInternetGraphDownloader()
    graphs = downloader.download_all_datasets()
    
    if not graphs:
        print("❌ Không thể tải datasets. Kết thúc benchmark.")
        return
    
    print(f"✅ Đã tải thành công {len(graphs)} datasets")
    
    # Step 2: Show dataset statistics
    print("\n📊 STEP 2: DATASET STATISTICS")
    stats_df = downloader.get_dataset_stats(graphs)
    print(stats_df.to_string(index=False))
    
    # Step 3: Run comprehensive benchmark
    print("\n⚡ STEP 3: RUNNING COMPREHENSIVE BENCHMARK")
    print("Algorithms: MAGS, MAGS-DM, Greedy Baseline, LDME Approximation")
    print("Metrics: Compression Ratio, Runtime, Query Efficiency")
    
    benchmark = ComprehensiveBenchmark()
    
    start_time = time.time()
    results_df = benchmark.run_comprehensive_benchmark(graphs)
    total_time = time.time() - start_time
    
    print(f"\n✅ Benchmark hoàn thành trong {total_time:.2f} giây")
    
    # Step 4: Create visualizations and analysis
    print("\n📈 STEP 4: CREATING ANALYSIS AND VISUALIZATIONS")
    create_visualization(results_df)
    
    # Step 5: Display key results
    print("\n🏆 STEP 5: KEY RESULTS SUMMARY")
    
    algorithms = ['MAGS', 'MAGS-DM', 'Greedy', 'LDME']
    
    print(f"\n{'Dataset':<15} {'Nodes':<8} {'Edges':<8}", end="")
    for alg in algorithms:
        print(f" {alg:<12}", end="")
    print()
    print("-" * (15 + 8 + 8 + 12 * len(algorithms)))
    
    for _, row in results_df.iterrows():
        print(f"{row['dataset']:<15} {row['original_nodes']:<8} {row['original_edges']:<8}", end="")
        for alg in algorithms:
            compression_col = f'{alg}_compression_ratio'
            if compression_col in row:
                print(f" {row[compression_col]:<11.1f}%", end="")
            else:
                print(f" {'N/A':<12}", end="")
        print()
    
    # Step 6: Best algorithm analysis
    print(f"\n🥇 STEP 6: BEST ALGORITHM ANALYSIS")
    
    best_compression = {}
    best_runtime = {}
    best_query_speedup = {}
    
    for _, row in results_df.iterrows():
        dataset = row['dataset']
        
        # Best compression
        best_comp_ratio = 0
        best_comp_alg = None
        for alg in algorithms:
            col = f'{alg}_compression_ratio'
            if col in row and row[col] > best_comp_ratio:
                best_comp_ratio = row[col]
                best_comp_alg = alg
        best_compression[dataset] = (best_comp_alg, best_comp_ratio)
        
        # Best runtime
        best_time = float('inf')
        best_time_alg = None
        for alg in algorithms:
            col = f'{alg}_runtime'
            if col in row and row[col] < best_time:
                best_time = row[col]
                best_time_alg = alg
        best_runtime[dataset] = (best_time_alg, best_time)
        
        # Best query speedup
        best_speedup = 0
        best_speedup_alg = None
        for alg in algorithms:
            col = f'{alg}_query_speedup'
            if col in row and row[col] > best_speedup and row[col] != float('inf'):
                best_speedup = row[col]
                best_speedup_alg = alg
        best_query_speedup[dataset] = (best_speedup_alg, best_speedup)
    
    print("\nBest Compression Ratio:")
    for dataset, (alg, ratio) in best_compression.items():
        if alg:
            print(f"  {dataset}: {alg} ({ratio:.1f}%)")
    
    print("\nBest Runtime:")
    for dataset, (alg, runtime) in best_runtime.items():
        if alg and runtime != float('inf'):
            print(f"  {dataset}: {alg} ({runtime:.3f}s)")
    
    print("\nBest Query Speedup:")
    for dataset, (alg, speedup) in best_query_speedup.items():
        if alg and speedup > 0:
            print(f"  {dataset}: {alg} ({speedup:.2f}x)")
    
    print(f"\n🎯 BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved in: benchmark_results/")
    print(f"📊 Detailed CSV: benchmark_results/detailed_results.csv")
    print(f"📈 Visualizations: benchmark_results/benchmark_comparison.png")

if __name__ == "__main__":
    main() 