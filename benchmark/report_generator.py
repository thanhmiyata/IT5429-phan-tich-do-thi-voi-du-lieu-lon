#!/usr/bin/env python3
"""
Report Generator cho Graph Summarization Benchmark
Tạo báo cáo markdown tự động từ kết quả benchmark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import yaml
import json
import argparse

class BenchmarkReportGenerator:
    """Tự động generate báo cáo từ kết quả benchmark"""
    
    def __init__(self, results_file: str, config_file: str = "config/config.yaml"):
        self.results_file = results_file
        self.config_file = config_file
        
        # Load data
        self.df = pd.read_csv(results_file)
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup matplotlib cho tiếng Việt
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS']
        
    def generate_report(self, output_file: str = None):
        """Generate báo cáo hoàn chỉnh"""
        
        if output_file is None:
            output_file = self.config['output']['report_file']
            
        report_content = self._generate_markdown_report()
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print(f"✅ Báo cáo đã được tạo: {output_file}")
        
        # Generate plots
        self._generate_plots()
        
    def _generate_markdown_report(self) -> str:
        """Generate nội dung markdown"""
        
        report = f"""# BÁO CÁO THỰC NGHIỆM GRAPH SUMMARIZATION

**Ngày tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

**Hệ thống benchmark:** Advanced Graph Summarization Benchmark System

---

## 1. TỔNG QUAN THỰC NGHIỆM

Báo cáo này trình bày kết quả thực nghiệm các thuật toán graph summarization trên các bộ dữ liệu đồ thị Internet, dựa trên nghiên cứu **"Graph Summarization: Compactness Meets Efficiency"** (SIGMOD 2024).

### 1.1. Thuật toán được đánh giá

{self._get_algorithms_summary()}

### 1.2. Bộ dữ liệu

{self._get_datasets_summary()}

### 1.3. Môi trường thực nghiệm

- **Hệ điều hành:** {self._get_system_info()}
- **Cấu hình phần cứng:** Multi-threading với {self._get_thread_count()} threads
- **Timeout:** {self.config['benchmark']['timeout']} giây cho mỗi thuật toán

---

## 2. KẾT QUẢ THỰC NGHIỆM

### 2.1. Hiệu năng Tóm tắt Đồ thị

{self._generate_summarization_performance_table()}

### 2.2. So sánh Thời gian Thực thi

{self._generate_execution_time_analysis()}

### 2.3. Chất lượng Tóm tắt (Compression Ratio)

{self._generate_compression_analysis()}

### 2.4. Hiệu năng Truy vấn trên Đồ thị Tóm tắt

{self._generate_query_performance_analysis()}

---

## 3. PHÂN TÍCH CHI TIẾT

### 3.1. Hiệu quả theo Dataset

{self._generate_dataset_analysis()}

### 3.2. Resource Usage Analysis

{self._generate_resource_analysis()}

---

## 4. KẾT LUẬN VÀ ĐÁNH GIÁ

{self._generate_conclusions()}

---

## 5. PHỤ LỤC

### 5.1. Cấu hình Chi tiết

```yaml
{yaml.dump(self.config, default_flow_style=False, allow_unicode=True)}
```

### 5.2. Raw Data Summary

- **Tổng số kết quả:** {len(self.df)}
- **Datasets được test:** {len(self.df['dataset'].unique())}
- **Algorithms được test:** {len(self.df[self.df['algorithm'] != 'Original']['algorithm'].unique())}

---

*Báo cáo được tạo tự động bởi Advanced Graph Summarization Benchmark System*
"""
        return report
        
    def _get_algorithms_summary(self) -> str:
        """Tóm tắt các thuật toán"""
        algorithms = self.df[self.df['algorithm'] != 'Original']['algorithm'].unique()
        
        algo_descriptions = {
            'mags': '**MAGS**: MinHash Assisted Graph Summarization - thuật toán chính từ paper',
            'mags_dm': '**MAGS-DM**: Divide-and-Merge variant của MAGS cho khả năng mở rộng tốt hơn',
            'ldme': '**LDME**: Thuật toán baseline để so sánh'
        }
        
        summary = []
        for algo in algorithms:
            if algo in algo_descriptions:
                summary.append(f"- {algo_descriptions[algo]}")
            else:
                summary.append(f"- **{algo.upper()}**: Thuật toán {algo}")
                
        return '\n'.join(summary)
        
    def _get_datasets_summary(self) -> str:
        """Tóm tắt datasets"""
        datasets = self.df['dataset'].unique()
        
        summary = []
        for dataset in datasets:
            if dataset in self.config['datasets']:
                config = self.config['datasets'][dataset]
                summary.append(f"- **{dataset}**: {config['description']} ({config['nodes']:,} nodes, {config['edges']:,} edges)")
            else:
                summary.append(f"- **{dataset}**: Dataset {dataset}")
                
        return '\n'.join(summary)
        
    def _get_system_info(self) -> str:
        """Thông tin hệ thống"""
        import platform
        return f"{platform.system()} {platform.release()}"
        
    def _get_thread_count(self) -> int:
        """Số threads được sử dụng"""
        return self.config['algorithms']['mags']['parameters']['threads']
        
    def _generate_summarization_performance_table(self) -> str:
        """Bảng hiệu năng tóm tắt"""
        
        # Filter summarization results
        summ_df = self.df[self.df['algorithm'] != 'Original'].copy()
        
        if summ_df.empty:
            return "*Không có dữ liệu tóm tắt*"
            
        # Create summary table
        pivot_time = summ_df.pivot_table(
            values='summarization_time_ms', 
            index='dataset', 
            columns='algorithm', 
            aggfunc='mean'
        ).round(0)
        
        pivot_compression = summ_df.pivot_table(
            values='summary_size_ratio', 
            index='dataset', 
            columns='algorithm', 
            aggfunc='mean'
        ).round(3)
        
        table_md = "#### Thời gian Tóm tắt (ms)\n\n"
        table_md += pivot_time.to_markdown() + "\n\n"
        
        table_md += "#### Tỷ lệ Nén (Compression Ratio)\n\n"
        table_md += pivot_compression.to_markdown() + "\n\n"
        
        return table_md
        
    def _generate_execution_time_analysis(self) -> str:
        """Phân tích thời gian thực thi"""
        
        summ_df = self.df[self.df['algorithm'] != 'Original'].copy()
        
        if summ_df.empty:
            return "*Không có dữ liệu phân tích*"
            
        # Statistics by algorithm
        stats = summ_df.groupby('algorithm')['summarization_time_ms'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(0)
        
        analysis = "#### Thống kê Thời gian Thực thi\n\n"
        analysis += stats.to_markdown() + "\n\n"
        
        # Find best performer
        best_algo = stats['mean'].idxmin()
        best_time = stats.loc[best_algo, 'mean']
        
        analysis += f"**Thuật toán nhanh nhất:** {best_algo.upper()} với thời gian trung bình {best_time:.0f}ms\n\n"
        
        return analysis
        
    def _generate_compression_analysis(self) -> str:
        """Phân tích chất lượng nén"""
        
        summ_df = self.df[self.df['algorithm'] != 'Original'].copy()
        
        if summ_df.empty:
            return "*Không có dữ liệu nén*"
            
        # Compression statistics
        comp_stats = summ_df.groupby('algorithm')['summary_size_ratio'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)
        
        analysis = "#### Thống kê Tỷ lệ Nén\n\n"
        analysis += comp_stats.to_markdown() + "\n\n"
        
        # Find best compressor
        best_compressor = comp_stats['mean'].idxmax()
        best_ratio = comp_stats.loc[best_compressor, 'mean']
        
        analysis += f"**Nén tốt nhất:** {best_compressor.upper()} với tỷ lệ nén trung bình {best_ratio:.1%}\n\n"
        
        return analysis
        
    def _generate_query_performance_analysis(self) -> str:
        """Phân tích hiệu năng truy vấn"""
        
        # Compare query performance on original vs summarized graphs
        original_df = self.df[self.df['algorithm'] == 'Original']
        summ_df = self.df[self.df['algorithm'] != 'Original']
        
        if original_df.empty or summ_df.empty:
            return "*Không đủ dữ liệu để so sánh*"
            
        analysis = "#### So sánh Hiệu năng Truy vấn\n\n"
        
        # PageRank analysis
        analysis += "**PageRank Performance:**\n\n"
        
        pr_comparison = []
        for dataset in original_df['dataset'].unique():
            orig_time = original_df[original_df['dataset'] == dataset]['pagerank_time_ms'].iloc[0]
            
            for algo in summ_df['algorithm'].unique():
                algo_data = summ_df[(summ_df['dataset'] == dataset) & (summ_df['algorithm'] == algo)]
                if not algo_data.empty:
                    summ_time = algo_data['pagerank_time_ms'].iloc[0]
                    accuracy = algo_data['pagerank_accuracy'].iloc[0]
                    speedup = orig_time / summ_time
                    
                    pr_comparison.append({
                        'Dataset': dataset,
                        'Algorithm': algo,
                        'Original Time (ms)': f"{orig_time:.0f}",
                        'Summary Time (ms)': f"{summ_time:.0f}",
                        'Speedup': f"{speedup:.1f}x",
                        'Accuracy': f"{accuracy:.1%}"
                    })
        
        if pr_comparison:
            pr_df = pd.DataFrame(pr_comparison)
            analysis += pr_df.to_markdown(index=False) + "\n\n"
            
        return analysis
        
    def _generate_dataset_analysis(self) -> str:
        """Phân tích theo dataset"""
        
        analysis = ""
        
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]
            
            analysis += f"#### {dataset}\n\n"
            
            if dataset in self.config['datasets']:
                config = self.config['datasets'][dataset]
                analysis += f"- **Mô tả:** {config['description']}\n"
                analysis += f"- **Kích thước:** {config['nodes']:,} nodes, {config['edges']:,} edges\n"
                analysis += f"- **Loại:** {config['type']}\n\n"
                
            # Performance summary for this dataset
            perf_summary = dataset_df[dataset_df['algorithm'] != 'Original'].groupby('algorithm').agg({
                'summarization_time_ms': 'mean',
                'summary_size_ratio': 'mean'
            }).round(2)
            
            if not perf_summary.empty:
                analysis += "**Hiệu năng trên dataset này:**\n\n"
                analysis += perf_summary.to_markdown() + "\n\n"
                
        return analysis
        
    def _generate_resource_analysis(self) -> str:
        """Phân tích resource usage"""
        
        summ_df = self.df[self.df['algorithm'] != 'Original'].copy()
        
        if summ_df.empty or 'memory_peak_mb' not in summ_df.columns:
            return "*Không có dữ liệu resource usage*"
            
        # Memory usage analysis
        memory_stats = summ_df.groupby('algorithm')['memory_peak_mb'].agg([
            'mean', 'std', 'max'
        ]).round(1)
        
        analysis = "#### Memory Usage (MB)\n\n"
        analysis += memory_stats.to_markdown() + "\n\n"
        
        # CPU usage analysis  
        if 'cpu_avg_percent' in summ_df.columns:
            cpu_stats = summ_df.groupby('algorithm')['cpu_avg_percent'].agg([
                'mean', 'std', 'max'
            ]).round(1)
            
            analysis += "#### CPU Usage (%)\n\n"
            analysis += cpu_stats.to_markdown() + "\n\n"
            
        return analysis
        
    def _generate_conclusions(self) -> str:
        """Generate kết luận"""
        
        summ_df = self.df[self.df['algorithm'] != 'Original'].copy()
        
        if summ_df.empty:
            return "*Không đủ dữ liệu để kết luận*"
            
        conclusions = []
        
        # Best algorithm by speed
        speed_ranking = summ_df.groupby('algorithm')['summarization_time_ms'].mean().sort_values()
        fastest = speed_ranking.index[0]
        conclusions.append(f"- **Tốc độ:** {fastest.upper()} là thuật toán nhanh nhất với thời gian trung bình {speed_ranking.iloc[0]:.0f}ms")
        
        # Best algorithm by compression
        comp_ranking = summ_df.groupby('algorithm')['summary_size_ratio'].mean().sort_values(ascending=False)
        best_comp = comp_ranking.index[0]
        conclusions.append(f"- **Chất lượng nén:** {best_comp.upper()} đạt tỷ lệ nén cao nhất {comp_ranking.iloc[0]:.1%}")
        
        # Query performance
        if 'pagerank_time_ms' in summ_df.columns:
            avg_speedup = []
            for dataset in self.df['dataset'].unique():
                orig_time = self.df[(self.df['dataset'] == dataset) & (self.df['algorithm'] == 'Original')]['pagerank_time_ms']
                if not orig_time.empty:
                    orig_time = orig_time.iloc[0]
                    summ_times = summ_df[summ_df['dataset'] == dataset]['pagerank_time_ms']
                    if not summ_times.empty:
                        avg_speedup.extend([orig_time / t for t in summ_times])
                        
            if avg_speedup:
                overall_speedup = np.mean(avg_speedup)
                conclusions.append(f"- **Hiệu năng truy vấn:** Trung bình nhanh hơn {overall_speedup:.1f}x so với đồ thị gốc")
                
        conclusions.append("- **Tổng kết:** Các thuật toán MAGS/MAGS-DM cho thấy hiệu quả tốt trong việc tóm tắt đồ thị Internet với thời gian hợp lý và chất lượng cao")
        
        return '\n'.join(conclusions)
        
    def _generate_plots(self):
        """Generate các biểu đồ minh họa"""
        
        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        summ_df = self.df[self.df['algorithm'] != 'Original'].copy()
        
        if summ_df.empty:
            return
            
        # 1. Execution time comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=summ_df, x='algorithm', y='summarization_time_ms')
        plt.title('So sánh Thời gian Thực thi các Thuật toán')
        plt.xlabel('Thuật toán')
        plt.ylabel('Thời gian (ms)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(plots_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Compression ratio comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=summ_df, x='algorithm', y='summary_size_ratio')
        plt.title('So sánh Tỷ lệ Nén các Thuật toán')
        plt.xlabel('Thuật toán')
        plt.ylabel('Tỷ lệ Nén')
        plt.tight_layout()
        plt.savefig(plots_dir / 'compression_ratio_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Biểu đồ đã được tạo trong thư mục {plots_dir}/")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--results", default="results/benchmark_results.csv",
                       help="Path to benchmark results CSV")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--output", help="Output report file")
    
    args = parser.parse_args()
    
    try:
        generator = BenchmarkReportGenerator(args.results, args.config)
        generator.generate_report(args.output)
        print("🎉 Báo cáo đã được tạo thành công!")
    except Exception as e:
        print(f"❌ Lỗi tạo báo cáo: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 