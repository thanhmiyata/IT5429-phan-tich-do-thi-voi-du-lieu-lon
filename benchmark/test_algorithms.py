#!/usr/bin/env python3
"""
Test script để kiểm tra algorithms và validate setup
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import time

def create_test_graph():
    """Tạo đồ thị test nhỏ"""
    test_graph = """# Test graph for algorithm validation
1 2
2 3
3 4
4 1
1 3
2 4
5 6
6 7
7 8
8 5
5 7
6 8
1 5
3 7
"""
    
    test_file = "temp/test_graph.txt"
    Path("temp").mkdir(exist_ok=True)
    
    with open(test_file, 'w') as f:
        f.write(test_graph)
        
    return test_file

def test_mags():
    """Test MAGS algorithm"""
    print("🧪 Testing MAGS algorithm...")
    
    test_file = create_test_graph()
    output_file = "temp/test_mags.summary"
    
    try:
        cmd = ["src/mags", test_file, output_file, "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✅ MAGS test passed")
            print(f"  📊 Output: {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ MAGS test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⏰ MAGS test timed out")
        return False
    except FileNotFoundError:
        print("  ❌ MAGS executable not found")
        return False

def test_mags_dm():
    """Test MAGS-DM algorithm"""
    print("🧪 Testing MAGS-DM algorithm...")
    
    test_file = create_test_graph()
    output_file = "temp/test_mags_dm.summary"
    
    try:
        cmd = ["src/mags_dm", test_file, output_file, "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✅ MAGS-DM test passed")
            print(f"  📊 Output: {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ MAGS-DM test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⏰ MAGS-DM test timed out")
        return False
    except FileNotFoundError:
        print("  ❌ MAGS-DM executable not found")
        return False

def test_python_imports():
    """Test Python dependencies"""
    print("🧪 Testing Python imports...")
    
    imports_to_test = [
        'yaml', 'pandas', 'numpy', 'psutil', 
        'tqdm', 'matplotlib', 'seaborn'
    ]
    
    failed_imports = []
    for module in imports_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed_imports.append(module)
            
    return len(failed_imports) == 0

def test_config_loading():
    """Test config loading"""
    print("🧪 Testing config loading...")
    
    try:
        import yaml
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("  ✅ Config loaded successfully")
        
        # Validate required sections
        required_sections = ['datasets', 'algorithms', 'benchmark', 'output']
        for section in required_sections:
            if section not in config:
                print(f"  ❌ Missing config section: {section}")
                return False
            else:
                print(f"  ✅ {section} section found")
                
        return True
        
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Running Algorithm Tests")
    print("=" * 50)
    
    tests = [
        ("Python Imports", test_python_imports),
        ("Config Loading", test_config_loading),
        ("MAGS Algorithm", test_mags),
        ("MAGS-DM Algorithm", test_mags_dm),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"  ⚠️  {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System ready for benchmark.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before running benchmark.")
        return 1

if __name__ == "__main__":
    exit(main()) 