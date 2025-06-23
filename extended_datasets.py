#!/usr/bin/env python3
"""
Mở rộng dataset downloader để có 10 internet graph datasets
Bao gồm: AS topology, Web graphs, P2P networks, Social networks liên quan internet
"""

import os
import requests
import gzip
import zipfile
import networkx as nx
from typing import Dict, List, Tuple
import pandas as pd

class ExtendedInternetGraphDownloader:
    def __init__(self, data_dir: str = "extended_graph_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 10 Internet-related graph datasets
        self.datasets = {
            # AS Topology Networks
            "caida_as": {
                "url": "https://snap.stanford.edu/data/as-caida20071105.txt.gz",
                "filename": "as-caida20071105.txt.gz",
                "description": "CAIDA AS Relationships",
                "type": "AS_topology"
            },
            "skitter_as": {
                "url": "https://snap.stanford.edu/data/as-skitter.txt.gz", 
                "filename": "as-skitter.txt.gz",
                "description": "Skitter AS Graph",
                "type": "AS_topology"
            },
            
            # Web Graphs
            "stanford_web": {
                "url": "https://snap.stanford.edu/data/web-Stanford.txt.gz",
                "filename": "web-Stanford.txt.gz", 
                "description": "Stanford Web Graph",
                "type": "web_graph"
            },
            "berkeley_web": {
                "url": "https://snap.stanford.edu/data/web-BerkStan.txt.gz",
                "filename": "web-BerkStan.txt.gz",
                "description": "Berkeley-Stanford Web Graph", 
                "type": "web_graph"
            },
            "google_web": {
                "url": "https://snap.stanford.edu/data/web-Google.txt.gz",
                "filename": "web-Google.txt.gz",
                "description": "Google Web Graph",
                "type": "web_graph"
            },
            
            # P2P Networks
            "gnutella_p2p": {
                "url": "https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz",
                "filename": "p2p-Gnutella31.txt.gz",
                "description": "Gnutella P2P Network",
                "type": "p2p_network"
            },
            "gnutella04_p2p": {
                "url": "https://snap.stanford.edu/data/p2p-Gnutella04.txt.gz", 
                "filename": "p2p-Gnutella04.txt.gz",
                "description": "Gnutella P2P Network (Aug 2002)",
                "type": "p2p_network"
            },
            
            # Email Networks (Internet-based communication)
            "email_enron": {
                "url": "https://snap.stanford.edu/data/email-Enron.txt.gz",
                "filename": "email-Enron.txt.gz", 
                "description": "Enron Email Network",
                "type": "email_network"
            },
            
            # Internet Infrastructure
            "router_level": {
                "url": "https://snap.stanford.edu/data/roadNet-PA.txt.gz",
                "filename": "roadNet-PA.txt.gz",
                "description": "Pennsylvania Road Network (Internet routing analogy)",
                "type": "infrastructure"
            },
            
            # Citation Network (Academic Internet)
            "arxiv_hepth": {
                "url": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
                "filename": "cit-HepTh.txt.gz",
                "description": "ArXiv HEP-TH Citation Network",
                "type": "citation_network"
            }
        }
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download file từ URL"""
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} đã tồn tại, bỏ qua download")
            return True
            
        try:
            print(f"Đang tải {filename} từ {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Đã tải xong {filename}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi tải {filename}: {e}")
            return False
    
    def extract_file(self, filename: str) -> str:
        """Giải nén file và trả về tên file text"""
        filepath = os.path.join(self.data_dir, filename)
        
        if filename.endswith('.gz'):
            output_filename = filename[:-3]  # Remove .gz
            output_filepath = os.path.join(self.data_dir, output_filename)
            
            if os.path.exists(output_filepath):
                return output_filename
                
            try:
                with gzip.open(filepath, 'rb') as f_in:
                    with open(output_filepath, 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f"Giải nén {filename}")
                return output_filename
            except Exception as e:
                print(f"Lỗi giải nén {filename}: {e}")
                return None
                
        elif filename.endswith('.zip'):
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print(f"Giải nén {filename}")
                return filename[:-4] + '.txt'  # Assume .txt inside
            except Exception as e:
                print(f"Lỗi giải nén {filename}: {e}")
                return None
        
        return filename
    
    def parse_snap_format(self, filename: str) -> nx.Graph:
        """Parse SNAP format graph file"""
        filepath = os.path.join(self.data_dir, filename)
        G = nx.Graph()
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            u, v = int(parts[0]), int(parts[1])
                            G.add_edge(u, v)
                        except ValueError:
                            continue
            
            return G
        except Exception as e:
            print(f"Lỗi parse file {filename}: {e}")
            return None
    
    def download_all_datasets(self) -> Dict[str, nx.Graph]:
        """Download và load tất cả datasets"""
        graphs = {}
        
        print("=== DOWNLOADING 10 INTERNET GRAPH DATASETS ===\n")
        
        for dataset_name, info in self.datasets.items():
            print(f"--- Processing {dataset_name.upper()} ---")
            print(f"Description: {info['description']}")
            print(f"Type: {info['type']}")
            
            # Download
            success = self.download_file(info['url'], info['filename'])
            if not success:
                print(f"Bỏ qua {dataset_name} do lỗi download\n")
                continue
            
            # Extract
            extracted_filename = self.extract_file(info['filename'])
            if not extracted_filename:
                print(f"Bỏ qua {dataset_name} do lỗi giải nén\n")
                continue
            
            # Parse
            graph = self.parse_snap_format(extracted_filename)
            if graph is None:
                print(f"Bỏ qua {dataset_name} do lỗi parse\n")
                continue
            
            # Remove self-loops and ensure undirected
            graph.remove_edges_from(nx.selfloop_edges(graph))
            
            print(f"Loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            graphs[dataset_name] = graph
            print()
        
        return graphs
    
    def get_dataset_stats(self, graphs: Dict[str, nx.Graph]) -> pd.DataFrame:
        """Tạo bảng thống kê các datasets"""
        stats = []
        
        for name, graph in graphs.items():
            info = self.datasets[name]
            
            # Basic stats
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            
            # Degree stats
            degrees = [d for n, d in graph.degree()]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            max_degree = max(degrees) if degrees else 0
            
            # Density
            density = nx.density(graph)
            
            # Connected components
            n_components = nx.number_connected_components(graph)
            largest_cc_size = len(max(nx.connected_components(graph), key=len))
            
            stats.append({
                'Dataset': name,
                'Description': info['description'], 
                'Type': info['type'],
                'Nodes': n_nodes,
                'Edges': n_edges,
                'Avg_Degree': round(avg_degree, 2),
                'Max_Degree': max_degree,
                'Density': round(density, 6),
                'Components': n_components,
                'Largest_CC': largest_cc_size,
                'CC_Ratio': round(largest_cc_size / n_nodes, 3) if n_nodes > 0 else 0
            })
        
        return pd.DataFrame(stats)

def main():
    """Test download tất cả datasets"""
    downloader = ExtendedInternetGraphDownloader()
    
    # Download all datasets
    graphs = downloader.download_all_datasets()
    
    print(f"=== THÀNH CÔNG TẢI {len(graphs)} DATASETS ===")
    
    # Show statistics
    if graphs:
        stats_df = downloader.get_dataset_stats(graphs)
        print("\nTHỐNG KÊ CÁC DATASETS:")
        print("=" * 120)
        print(stats_df.to_string(index=False))
        
        # Save stats to CSV
        stats_df.to_csv('extended_graph_data/dataset_statistics.csv', index=False)
        print(f"\nĐã lưu thống kê vào extended_graph_data/dataset_statistics.csv")
    
    return graphs

if __name__ == "__main__":
    main() 