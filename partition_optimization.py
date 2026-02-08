"""
partition_optimization.py

Partition optimization utilities for efficient large-file processing.
Tests different partition sizes to find optimal balance between memory use and performance.
Uses PyArrow batching and memory profiling to select best partition size.
"""

import re
import time
import psutil
import os
from typing import Optional, Tuple, Dict, Any, List
import logging
import pyarrow.parquet as pq
import dask.dataframe as dd

logger = logging.getLogger(__name__)


# ============================================================================
# Size Parsing
# ============================================================================


def parse_size(size_str: str) -> int:
    """
    Parse a human-readable size string to bytes.
    
    Args:
        size_str: Size string (e.g., "200MB", "1.5GB", "1024KB", "2TB")
        
    Returns:
        Size in bytes as an integer
        
    Raises:
        ValueError: If size_str format is invalid
        
    Examples:
        - "200MB" → 209715200
        - "1.5GB" → 1610612736
        - "50MB" → 52428800
        - "1024KB" → 1048576
        - "2TB" → 2199023255552
    """
    if not size_str or not isinstance(size_str, str):
        raise ValueError(f"size_str must be a non-empty string, got {size_str}")
    
    size_str = size_str.strip().upper()
    
    # Pattern: number (int or float) followed by unit
    match = re.match(r'^([\d.]+)\s*([KMGT]B|B)$', size_str)
    
    if not match:
        raise ValueError(
            f"Invalid size format: {size_str}. Expected format like '200MB', '1.5GB', etc."
        )
    
    value_str, unit = match.groups()
    
    try:
        value = float(value_str)
    except ValueError:
        raise ValueError(f"Could not parse numeric value: {value_str}")
    
    # Unit multipliers
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
    }
    
    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")
    
    result = int(value * units[unit])
    
    if result <= 0:
        raise ValueError(f"Size must be positive, got {result}")
    
    return result


def format_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string (e.g., "1.5GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            if unit == 'B':
                return f"{size_bytes}{unit}"
            else:
                return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.2f}TB"


# ============================================================================
# Memory Profiling
# ============================================================================


def get_process_memory_mb() -> float:
    """
    Get current process memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_available_memory_mb() -> float:
    """
    Get available system memory in MB.
    
    Returns:
        Available memory in MB
    """
    return psutil.virtual_memory().available / (1024 ** 2)


# ============================================================================
# Partition Optimization
# ============================================================================


def find_optimal_partition_size(
    file_path: str,
    candidate_sizes: Optional[List[int]] = None,
    size_range: Optional[Tuple[str, str]] = None,
    num_sizes: int = 5,
    max_memory_usage: Optional[int] = None,
    warmup: bool = True,
) -> Dict[str, Any]:
    """
    Find optimal partition size for a parquet file using PyArrow batching.
    
    Tests different partition sizes and measures time/memory for each,
    selecting the best size that fits within max_memory_usage constraint.
    
    Args:
        file_path: Path to parquet file (local or S3)
        candidate_sizes: List of partition sizes (bytes) to test.
                        If None, generates sizes from size_range or defaults (50MB-1GB).
        size_range: Tuple of (min_size_str, max_size_str) to generate candidates
                   (e.g., ("50MB", "1GB")). Used only if candidate_sizes is None.
        num_sizes: Number of candidate sizes to generate if using size_range (default 5)
        max_memory_usage: Max memory in bytes to allow; suggest sizes within this limit.
                         If None, uses 80% of available system memory.
        warmup: Whether to run a warmup iteration first (default True)
        
    Returns:
        Dictionary with keys:
            - 'optimal_size': Recommended partition size in bytes
            - 'optimal_size_str': Human-readable format
            - 'results': List of test results for each candidate
            - 'max_memory_usage': Constraint used
            - 'file_info': Info about the test file
            
    Raises:
        ValueError: If file doesn't exist or candidate_sizes is empty
        
    Example:
        >>> result = find_optimal_partition_size(
        ...     'data.parquet',
        ...     size_range=("50MB", "1GB"),
        ...     max_memory_usage=parse_size("20GB")
        ... )
        >>> print(f"Optimal size: {result['optimal_size_str']}")
    """
    
    # Validate file exists
    if not file_path:
        raise ValueError("file_path cannot be empty")
    
    # Get file size
    try:
        pf = pq.ParquetFile(file_path)
        file_size_bytes = pf.metadata.serialized_size
    except Exception as e:
        raise ValueError(f"Could not read parquet file {file_path}: {e}")
    
    logger.info(f"Testing partition sizes for {file_path} ({format_size(file_size_bytes)})")
    
    # Generate candidate sizes if not provided
    if candidate_sizes is None:
        if size_range is not None:
            min_size_str, max_size_str = size_range
            min_size = parse_size(min_size_str)
            max_size = parse_size(max_size_str)
        else:
            # Default: 50MB to 1GB
            min_size = parse_size("50MB")
            max_size = parse_size("1GB")
        
        # Generate logarithmically spaced candidates
        candidate_sizes = [
            int(min_size * (max_size / min_size) ** (i / (num_sizes - 1)))
            for i in range(num_sizes)
        ]
    
    if not candidate_sizes:
        raise ValueError("candidate_sizes cannot be empty")
    
    # Set memory constraint
    if max_memory_usage is None:
        available = get_available_memory_mb()
        max_memory_usage = int(available * 0.8 * 1024 * 1024)  # 80% of available
    
    logger.info(f"Max memory constraint: {format_size(max_memory_usage)}")
    
    # Warmup
    if warmup:
        logger.info("Running warmup iteration...")
        try:
            _ = dd.read_parquet(file_path, blocksize=candidate_sizes[0])
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    # Test each candidate size
    results = []
    baseline_memory = get_process_memory_mb()
    
    for i, partition_size in enumerate(candidate_sizes):
        logger.info(f"Testing partition size {i+1}/{len(candidate_sizes)}: {format_size(partition_size)}")
        
        test_result = {
            'partition_size': partition_size,
            'partition_size_str': format_size(partition_size),
        }
        
        try:
            # Record start time and memory
            start_time = time.time()
            start_memory = get_process_memory_mb()
            
            # Read with dask using blocksize (partition size)
            df = dd.read_parquet(file_path, blocksize=partition_size)
            
            # Force computation to actually load and process data
            _ = df.head(1)
            
            # Record end time and memory
            end_time = time.time()
            end_memory = get_process_memory_mb()
            
            elapsed_time = end_time - start_time
            memory_used = end_memory - start_memory
            peak_memory = end_memory - baseline_memory
            
            test_result.update({
                'elapsed_time_sec': elapsed_time,
                'memory_used_mb': memory_used,
                'peak_memory_mb': peak_memory,
                'success': True,
                'error': None,
            })
            
            logger.info(
                f"  Time: {elapsed_time:.2f}s | Memory: {memory_used:.1f}MB | "
                f"Within limit: {peak_memory < (max_memory_usage / (1024**2))}"
            )
            
        except Exception as e:
            logger.error(f"Error testing partition size {format_size(partition_size)}: {e}")
            test_result.update({
                'elapsed_time_sec': None,
                'memory_used_mb': None,
                'peak_memory_mb': None,
                'success': False,
                'error': str(e),
            })
        
        results.append(test_result)
    
    # Select optimal size
    # Criteria: successful, within memory limit, balanced time/memory
    optimal_result = None
    optimal_size = candidate_sizes[0]  # Fallback to first
    
    valid_results = [
        r for r in results
        if r['success'] and r['peak_memory_mb'] < (max_memory_usage / (1024**2))
    ]
    
    if valid_results:
        # Choose the one with lowest peak memory among valid results
        optimal_result = min(valid_results, key=lambda r: r['peak_memory_mb'])
        optimal_size = optimal_result['partition_size']
    elif results:
        # If no results within memory limit, choose fastest (best effort)
        successful_results = [r for r in results if r['success']]
        if successful_results:
            optimal_result = min(successful_results, key=lambda r: r['elapsed_time_sec'])
            optimal_size = optimal_result['partition_size']
            logger.warning(
                f"No partition size within memory limit; selecting fastest: "
                f"{optimal_result['partition_size_str']}"
            )
    
    return {
        'optimal_size': optimal_size,
        'optimal_size_str': format_size(optimal_size),
        'results': results,
        'max_memory_usage': max_memory_usage,
        'max_memory_usage_str': format_size(max_memory_usage),
        'file_info': {
            'path': file_path,
            'size_bytes': file_size_bytes,
            'size_str': format_size(file_size_bytes),
        },
    }


def test_partition_size(
    file_path: str,
    partition_size: int,
    num_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Test performance of a specific partition size with multiple iterations.
    
    Args:
        file_path: Path to parquet file
        partition_size: Partition size in bytes to test
        num_iterations: Number of iterations to run (for averaging)
        
    Returns:
        Dictionary with timing and memory statistics
    """
    logger.info(
        f"Testing partition size {format_size(partition_size)} "
        f"({num_iterations} iterations)..."
    )
    
    times = []
    memories = []
    baseline_memory = get_process_memory_mb()
    
    for i in range(num_iterations):
        try:
            start_time = time.time()
            start_memory = get_process_memory_mb()
            
            df = dd.read_parquet(file_path, blocksize=partition_size)
            _ = df.head(1)
            
            elapsed = time.time() - start_time
            memory_delta = get_process_memory_mb() - start_memory
            
            times.append(elapsed)
            memories.append(memory_delta)
            
            logger.info(
                f"  Iteration {i+1}: {elapsed:.2f}s, {memory_delta:.1f}MB"
            )
            
        except Exception as e:
            logger.error(f"Error in iteration {i+1}: {e}")
            raise
    
    return {
        'partition_size': partition_size,
        'partition_size_str': format_size(partition_size),
        'iterations': num_iterations,
        'avg_time_sec': sum(times) / len(times),
        'min_time_sec': min(times),
        'max_time_sec': max(times),
        'avg_memory_mb': sum(memories) / len(memories),
        'max_memory_mb': max(memories),
    }

# ============================================================================
# Demo: test candidate sizes (50MB–1GB), measure time/memory, pick best within max_memory_usage
# ============================================================================

if __name__ == "__main__":
    import sys

    # Default: one S3 taxi parquet (or pass a local path: python partition_optimization.py /path/to/file.parquet)
    DEFAULT_FILE = "s3://dsc291-ucsd/taxi/Dataset/2021/yellow_taxi/yellow_tripdata_2021-01.parquet"
    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

    # Optional: max memory cap (e.g. 2GB). If not set, script uses 80% of available.
    MAX_MEMORY_STR = "2GB"
    max_memory_usage = parse_size(MAX_MEMORY_STR)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=== Partition size optimization demo ===\n")
    print(f"File: {file_path}")
    print(f"Candidate range: 50MB – 1GB (log-spaced)")
    print(f"Max memory: {MAX_MEMORY_STR}\n")

    result = find_optimal_partition_size(
        file_path,
        size_range=("50MB", "1GB"),
        num_sizes=6,
        max_memory_usage=max_memory_usage,
        warmup=True,
    )

    print("\n--- Results ---")
    max_mb = result["max_memory_usage"] / (1024 ** 2)
    for r in result["results"]:
        status = "ok" if r["success"] and r["peak_memory_mb"] < max_mb else ("fail" if not r["success"] else "over")
        t = r["elapsed_time_sec"]
        m = r["peak_memory_mb"]
        err = f" | {r['error']}" if r.get("error") else ""
        print(f"  {r['partition_size_str']:>8}  time={t:.2f}s  peak_mem={m:.1f}MB  [{status}]{err}")
    print(f"\nOptimal (within limit): {result['optimal_size_str']}\n")