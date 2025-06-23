#!/usr/bin/env python3
"""
Script để tải và chuẩn bị các đồ thị internet cho thực nghiệm Graph Summarization
Dựa trên paper SIGMOD 2024: "Graph Summarization: Compactness Meets Efficiency"
"""

import os
import requests
import numpy as np
import networkx as nx
from urllib.parse import urljoin
import gzip
import tarfile
import zipfile
from pathlib import Path

class InternetGraphDownloader:
    def __init__(self, data_dir="./graph_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # URLs của các dataset
        self.graph_urls = {
            # SNAP datasets
            "caida": "https://snap.stanford.edu/data/as-caida20071105.txt.gz",
            "web_stanford": "https://snap.stanford.edu/data/web-Stanford.txt.gz",
            
            # Network Repository datasets  
            "cnr_2000": "http://networkrepository.com/cnr-2000.zip",
            "in_2004": "http://networkrepository.com/in-2004.zip",
            
            # LAW datasets (Large-scale Web graph datasets)
            "eu_2005": "http://law.di.unimi.it/webdata/eu-2005/eu-2005.graph",
            "uk_2005": "http://law.di.unimi.it/webdata/uk-2005/uk-2005.graph",
            "it_2004": "http://law.di.unimi.it/webdata/it-2004/it-2004.graph",
            
            # Skitter dataset
            "skitter": "https://snap.stanford.edu/data/as-skitter.txt.gz"
        }
    
    def download_file(self, url, filename):
        """Tải file từ URL"""
        filepath = self.data_dir / filename
        if filepath.exists():
            print(f"File {filename} đã tồn tại, bỏ qua tải xuống")
            return filepath
            
        print(f"Đang tải {filename} từ {url}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Đã tải xong {filename}")
            return filepath
        except Exception as e:
            print(f"Lỗi khi tải {filename}: {e}")
            return None
    
    def extract_compressed_file(self, filepath):
        """Giải nén file nếu cần"""
        if filepath.suffix == '.gz':
            output_path = filepath.with_suffix('')
            if not output_path.exists():
                print(f"Giải nén {filepath}")
                with gzip.open(filepath, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        f_out.write(f_in.read())
            return output_path
        elif filepath.suffix == '.zip':
            extract_dir = filepath.parent / filepath.stem
            if not extract_dir.exists():
                print(f"Giải nén {filepath}")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            return extract_dir
        return filepath
    
    def load_snap_format(self, filepath):
        """Đọc đồ thị định dạng SNAP"""
        edges = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        edges.append((u, v))
                    except ValueError:
                        continue
        
        G = nx.Graph()
        G.add_edges_from(edges)
        return G
    
    def download_and_prepare_datasets(self):
        """Tải và chuẩn bị tất cả datasets"""
        datasets = {}
        
        # CAIDA AS-level topology
        print("\n=== CAIDA AS Dataset ===")
        caida_file = self.download_file(self.graph_urls["caida"], "as-caida20071105.txt.gz")
        if caida_file:
            caida_extracted = self.extract_compressed_file(caida_file)
            datasets['caida'] = self.load_snap_format(caida_extracted)
            print(f"CAIDA: {datasets['caida'].number_of_nodes()} nodes, {datasets['caida'].number_of_edges()} edges")
        
        # Skitter AS-level topology  
        print("\n=== Skitter AS Dataset ===")
        skitter_file = self.download_file(self.graph_urls["skitter"], "as-skitter.txt.gz")
        if skitter_file:
            skitter_extracted = self.extract_compressed_file(skitter_file)
            datasets['skitter'] = self.load_snap_format(skitter_extracted)
            print(f"Skitter: {datasets['skitter'].number_of_nodes()} nodes, {datasets['skitter'].number_of_edges()} edges")
        
        # Stanford Web
        print("\n=== Stanford Web Dataset ===")
        stanford_file = self.download_file(self.graph_urls["web_stanford"], "web-Stanford.txt.gz")
        if stanford_file:
            stanford_extracted = self.extract_compressed_file(stanford_file)
            datasets['stanford_web'] = self.load_snap_format(stanford_extracted)
            print(f"Stanford Web: {datasets['stanford_web'].number_of_nodes()} nodes, {datasets['stanford_web'].number_of_edges()} edges")
        
        return datasets
    
    def save_graphs(self, datasets, format='edgelist'):
        """Lưu đồ thị theo định dạng được chỉ định"""
        output_dir = self.data_dir / "processed"
        output_dir.mkdir(exist_ok=True)
        
        for name, graph in datasets.items():
            if format == 'edgelist':
                output_file = output_dir / f"{name}.edges"
                nx.write_edgelist(graph, output_file, data=False)
            elif format == 'gml':
                output_file = output_dir / f"{name}.gml"
                nx.write_gml(graph, output_file)
            elif format == 'graphml':
                output_file = output_dir / f"{name}.graphml"
                nx.write_graphml(graph, output_file)
            
            print(f"Đã lưu {name} vào {output_file}")
    
    def get_graph_statistics(self, datasets):
        """In thống kê các đồ thị"""
        print("\n=== THỐNG KÊ CÁC ĐỒ THỊ ===")
        print(f"{'Dataset':<15} {'Nodes':<10} {'Edges':<12} {'Avg Degree':<12} {'Density':<10}")
        print("-" * 65)
        
        for name, graph in datasets.items():
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
            density = nx.density(graph)
            
            print(f"{name:<15} {n_nodes:<10} {n_edges:<12} {avg_degree:<12.2f} {density:<10.6f}")

def main():
    # Tạo thư mục làm việc
    downloader = InternetGraphDownloader()
    
    print("Bắt đầu tải các đồ thị internet cho thực nghiệm Graph Summarization...")
    
    # Tải và chuẩn bị datasets
    datasets = downloader.download_and_prepare_datasets()
    
    # In thống kê
    if datasets:
        downloader.get_graph_statistics(datasets)
        
        # Lưu đồ thị
        downloader.save_graphs(datasets, format='edgelist')
        
        print(f"\nĐã hoàn thành! Các file đã được lưu trong thư mục: {downloader.data_dir}")
        print("\nCác dataset này có thể sử dụng để thực nghiệm với thuật toán MAGS/MAGS-DM")
    else:
        print("Không thể tải dataset nào!")

if __name__ == "__main__":
    main() 