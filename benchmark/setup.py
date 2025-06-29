#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
from pathlib import Path

def main():
    print("🚀 Graph Summarization Benchmark Setup")
    
    # Install Python dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Create directories
    for d in ["results", "summaries", "logs", "build", "temp"]:
        Path(d).mkdir(exist_ok=True)
    
    # Compile C++
    subprocess.run(["make", "all"], check=True)
    
    print("✅ Setup hoàn thành!")

if __name__ == "__main__":
    main() 