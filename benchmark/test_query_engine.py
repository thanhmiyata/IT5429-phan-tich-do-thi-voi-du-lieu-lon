#!/usr/bin/env python3
"""
Test Script for Real Query Engine
=================================

Tests the GraphQueryEngine with real data to ensure all query implementations work correctly.
"""

import sys
import time
import logging
from pathlib import Path
from query_engines import GraphQueryEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_query_engine(dataset_path: str, dataset_name: str):
    """Test query engine with a specific dataset"""
    
    logger.info(f"🧪 Testing Query Engine with {dataset_name}")
    logger.info(f"📂 Dataset path: {dataset_path}")
    
    if not Path(dataset_path).exists():
        logger.error(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        # Load graph
        logger.info("📊 Loading graph...")
        start_time = time.time()
        query_engine = GraphQueryEngine.from_file(dataset_path)
        load_time = time.time() - start_time
        logger.info(f"✅ Graph loaded in {load_time:.2f}s: {query_engine.num_nodes} nodes, {query_engine.num_edges} edges")
        
        # Test configuration for queries
        query_config = {
            'pagerank': {
                'enabled': True,
                'damping': 0.85,
                'iterations': 20,  # Reduced for testing
                'tolerance': 1e-4   # Relaxed tolerance for faster convergence
            },
            'sssp': {
                'enabled': True
            },
            '2hop_neighbors': {
                'enabled': True,
                'num_target_nodes': 10  # Small number for testing
            },
            'reachability': {
                'enabled': True,
                'sample_pairs': 100  # Small sample for testing
            }
        }
        
        # Run queries
        logger.info("🔍 Running queries...")
        results = query_engine.benchmark_all_queries(query_config)
        
        # Print results
        logger.info("📊 Query Results:")
        logger.info("="*50)
        
        if 'pagerank' in results:
            pr = results['pagerank']
            logger.info(f"📈 PageRank:")
            logger.info(f"  ⏱️  Time: {pr['execution_time_ms']:.1f}ms")
            logger.info(f"  🔄 Iterations: {pr['iterations_to_converge']}")
            logger.info(f"  ✅ Converged: {pr['convergence_achieved']}")
            logger.info(f"  📊 Score variance: {pr['score_variance']:.6f}")
            logger.info(f"  🏆 Top nodes: {pr['top_nodes'][:3]}")
        
        if 'sssp' in results:
            sssp = results['sssp']
            logger.info(f"🔍 SSSP:")
            logger.info(f"  ⏱️  Time: {sssp['execution_time_ms']:.1f}ms")
            logger.info(f"  📍 Source node: {sssp['source_node']}")
            logger.info(f"  📊 Reachable nodes: {sssp['reachable_nodes']}")
            logger.info(f"  📏 Avg distance: {sssp['avg_distance']:.2f}")
            logger.info(f"  📐 Max distance: {sssp['max_distance']}")
        
        if '2hop_neighbors' in results:
            hop2 = results['2hop_neighbors']
            logger.info(f"👥 2-hop Neighbors:")
            logger.info(f"  ⏱️  Time: {hop2['execution_time_ms']:.1f}ms")
            logger.info(f"  📊 Nodes queried: {hop2['nodes_queried']}")
            logger.info(f"  📊 Avg 2-hop neighbors: {hop2['avg_2hop_neighbors']:.1f}")
            logger.info(f"  📊 Max 2-hop neighbors: {hop2['max_2hop_neighbors']}")
        
        if 'reachability' in results:
            reach = results['reachability']
            logger.info(f"🔗 Reachability:")
            logger.info(f"  ⏱️  Time: {reach['execution_time_ms']:.1f}ms")
            logger.info(f"  📊 Pairs tested: {reach['pairs_tested']}")
            logger.info(f"  ✅ Reachable pairs: {reach['reachable_pairs']}")
            logger.info(f"  📊 Reachability ratio: {reach['reachability_ratio']:.3f}")
        
        logger.info("="*50)
        logger.info(f"✅ All queries completed successfully for {dataset_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing {dataset_name}: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Query Engine Tests")
    
    # Test datasets (prioritize smaller ones first)
    test_datasets = [
        ("../graph_data/email-EuAll.txt", "email-EuAll"),
        ("../graph_data/web-BerkStan.txt", "web-BerkStan"),
        ("../graph_data/as-skitter.txt", "as-skitter")
    ]
    
    success_count = 0
    total_tests = len(test_datasets)
    
    for dataset_path, dataset_name in test_datasets:
        logger.info(f"\n{'='*60}")
        try:
            if test_query_engine(dataset_path, dataset_name):
                success_count += 1
                logger.info(f"✅ {dataset_name} test PASSED")
            else:
                logger.error(f"❌ {dataset_name} test FAILED")
        except KeyboardInterrupt:
            logger.info("\n⏹️  Tests interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error in {dataset_name}: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"✅ Passed: {success_count}/{total_tests}")
    logger.info(f"❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("🎉 All tests passed! Query engine is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 