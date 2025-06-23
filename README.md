# Thực nghiệm Graph Summarization với MAGS/MAGS-DM

Dự án này triển khai và thực nghiệm lại nghiên cứu từ paper SIGMOD 2024: **"Graph Summarization: Compactness Meets Efficiency"** của Chu et al., tập trung vào các đồ thị internet.

## Tổng quan về nghiên cứu

Paper đề xuất hai thuật toán graph summarization:
- **MAGS**: Thuật toán greedy cải tiến với độ phức tạp O(T·m·(d_avg + log m))
- **MAGS-DM**: Thuật toán divide-and-merge với độ phức tạp O(T·m)

## Các đồ thị Internet được sử dụng

### Internet Topology:
1. **CAIDA AS**: 26,475 nodes, 53,381 edges - Internet topology
2. **Skitter**: 1,696,415 nodes, 11,095,298 edges - AS-level topology

### Web Graphs:
3. **CNR-2000**: 325,557 nodes, 2,738,969 edges - Web crawl Italy
4. **IN-2004**: 1,382,867 nodes, 13,591,473 edges - Web crawl Indochina
5. **EU-2005**: 862,664 nodes, 16,138,468 edges - Web crawl Europe
6. **UK-2005**: 39,454,463 nodes, 783,027,125 edges - Web crawl UK
7. **IT-2004**: 41,290,648 nodes, 1,027,474,947 edges - Web crawl Italy

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Kết nối internet để tải datasets

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Cấu trúc thư mục
```
IT5429 - Phân tích đồ thị với dữ liệu lớn/
├── download_internet_graphs.py    # Script tải datasets
├── mags_implementation.py         # Triển khai thuật toán MAGS
├── requirements.txt               # Dependencies
├── README.md                      # Hướng dẫn này
└── graph_data/                    # Thư mục chứa datasets (tự động tạo)
    ├── processed/                 # Graphs đã xử lý
    └── ...
```

## Sử dụng

### 1. Tải và chuẩn bị datasets
```bash
python download_internet_graphs.py
```

Script này sẽ:
- Tải các dataset internet từ SNAP, Network Repository, LAW
- Giải nén và chuyển đổi về định dạng NetworkX
- Lưu vào thư mục `graph_data/`

### 2. Chạy thực nghiệm MAGS
```bash
python mags_implementation.py
```

Hoặc sử dụng trong code:
```python
from mags_implementation import MAGS
from download_internet_graphs import InternetGraphDownloader
import networkx as nx

# Tải dataset
downloader = InternetGraphDownloader()
datasets = downloader.download_and_prepare_datasets()

# Khởi tạo MAGS
mags = MAGS(k=5, T=50, b=5, h=10)

# Chạy summarization
for name, graph in datasets.items():
    print(f"Summarizing {name}...")
    super_nodes = mags.summarize(graph)
    stats = mags.get_summary_stats(graph, super_nodes)
    print(f"Compression ratio: {stats['compression_ratio']:.2f}%")
```

## Tham số thuật toán

### MAGS Parameters:
- `k`: Số candidate pairs cho mỗi node (default: 5)
- `T`: Số iterations (default: 50)
- `b`: Số nodes mẫu cho 2-hop neighbors (default: 5)  
- `h`: Số hash functions cho MinHash (default: 10)

### Tuning gợi ý:
- **Graphs nhỏ** (<10K nodes): T=50, k=10
- **Graphs trung bình** (10K-100K nodes): T=30, k=5
- **Graphs lớn** (>100K nodes): T=20, k=3

## Datasets chi tiết

### CAIDA AS-level Topology
- **Nguồn**: http://www.caida.org/
- **Mô tả**: Internet topology ở mức autonomous system
- **Ứng dụng**: Phân tích cấu trúc Internet, routing protocols

### Skitter Project
- **Nguồn**: https://www.caida.org/projects/skitter/
- **Mô tả**: AS-level topology từ traceroute measurements  
- **Ứng dụng**: Internet mapping, network analysis

### Web Crawl Datasets (LAW)
- **Nguồn**: http://law.di.unimi.it/
- **Mô tả**: Web graphs từ các crawls quy mô lớn
- **Ứng dụng**: Web structure analysis, PageRank computation

## Kết quả mong đợi

Theo paper gốc, MAGS đạt được:
- **Compactness**: Gần như tương đương với Greedy baseline (<0.1% chênh lệch)
- **Efficiency**: Nhanh hơn LDME 11.1x, Slugger 4.2x trung bình
- **Scalability**: Xử lý được graphs tỷ cạnh

### Compression ratios điển hình:
- Small graphs (1K-10K nodes): 20-40%
- Medium graphs (10K-100K nodes): 30-50%  
- Large graphs (>100K nodes): 40-60%

## Phân tích và Comparison

### So sánh với baselines:
1. **Greedy [Navlakha et al.]**: Compactness tốt nhất nhưng chậm
2. **LDME**: Nhanh nhưng kém compactness hơn 21.7%
3. **Slugger**: Nhanh nhưng kém compactness hơn 30.2%

### Metrics đánh giá:
- **Compression ratio**: (original_size - summary_size) / original_size  
- **Running time**: Thời gian thực hiện summarization
- **Summary quality**: Độ chính xác của summary graph

## Troubleshooting

### Lỗi thường gặp:
1. **Memory error**: Giảm kích thước graph test hoặc tăng RAM
2. **Download failed**: Kiểm tra kết nối internet, một số datasets có thể không available
3. **Import error**: Đảm bảo đã cài đặt đúng dependencies

### Giải pháp tối ưu:
- Với graphs lớn, sử dụng subgraph sampling
- Parallel processing cho multiple datasets
- Caching intermediate results

## Tham khảo

1. **Paper gốc**: Chu, D., et al. "Graph Summarization: Compactness Meets Efficiency." SIGMOD 2024.
2. **SNAP datasets**: http://snap.stanford.edu/data/
3. **Network Repository**: http://networkrepository.com/
4. **LAW datasets**: http://law.di.unimi.it/datasets.php

## License

Dự án này được phát triển cho mục đích học tập và nghiên cứu. Vui lòng tham khảo license của các datasets gốc khi sử dụng. 