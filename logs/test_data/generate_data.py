#!/usr/bin/env python3

import json
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Base directory
OUTPUT_DIR = "logs/test_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Configuration parameters
NODE_COUNTS = [3, 5, 7, 10, 15]  # Different numbers of nodes to simulate
PACKET_COUNTS = [100, 150, 200, 250, 300]  # Corresponding packets per node count
RUNS_PER_CONFIG = 5  # Number of runs per configuration

# Performance factors (how much CRDT improves/degrades performance)
# Values above 1.0 mean CRDT performs worse, below 1.0 means CRDT performs better
LATENCY_FACTOR = 1.15  # CRDT slightly increases latency
THROUGHPUT_FACTOR = 0.85  # CRDT improves throughput
SUCCESS_RATE_FACTOR = 0.7  # CRDT significantly improves success rate
MISSING_PACKETS_FACTOR = 0.6  # CRDT reduces missing packets

# Variance factors (how much randomness to introduce)
VARIANCE = 0.2  # 20% variance

def generate_latency_data(node_count, packet_count, use_crdt):
    """Generate random latency data in seconds"""
    # Base latency increases with node count (more hops = more delay)
    base_latency = 0.2 + (node_count * 0.03)
    
    # Apply CRDT factor if enabled
    if use_crdt:
        base_latency *= LATENCY_FACTOR
    
    # Generate random latencies with some variance
    latencies = np.random.normal(
        loc=base_latency, 
        scale=base_latency * VARIANCE, 
        size=packet_count
    )
    
    # Ensure no negative values
    return np.maximum(latencies, 0.05)

def generate_throughput_data(node_count, packet_count, use_crdt):
    """Generate random throughput data in packets/second"""
    # Base throughput decreases with node count (network contention)
    base_throughput = 5.0 - (node_count * 0.2)
    base_throughput = max(base_throughput, 1.0)  # Ensure minimum throughput
    
    # Apply CRDT factor if enabled
    if use_crdt:
        base_throughput /= THROUGHPUT_FACTOR  # CRDT improves throughput
    
    # Generate throughput measurements over time
    throughputs = np.random.normal(
        loc=base_throughput, 
        scale=base_throughput * VARIANCE, 
        size=min(30, packet_count // 3)  # Fewer throughput measurements than packets
    )
    
    return np.maximum(throughputs, 0.5)  # Ensure minimum throughput

def generate_success_rate(node_count, use_crdt):
    """Generate packet delivery success rate"""
    # Base success rate decreases with node count
    base_success_rate = 0.95 - (node_count * 0.03)
    
    # Apply CRDT factor if enabled (CRDT improves success rate)
    if use_crdt:
        base_success_rate = min(0.99, base_success_rate / SUCCESS_RATE_FACTOR)
    
    # Add some randomness
    success_rate = base_success_rate + random.uniform(-VARIANCE, VARIANCE) * base_success_rate
    
    # Clamp to valid range
    return max(0.1, min(0.99, success_rate))

def generate_missing_packets(packet_count, success_rate, use_crdt):
    """Calculate number of missing packets based on success rate"""
    expected_missing = int(packet_count * (1 - success_rate))
    
    # Apply CRDT factor if enabled
    if use_crdt:
        expected_missing = int(expected_missing * MISSING_PACKETS_FACTOR)
    
    # Add some randomness
    variance = int(expected_missing * VARIANCE)
    missing_packets = max(0, expected_missing + random.randint(-variance, variance))
    
    return missing_packets

def generate_hop_counts(node_count, packet_count, use_crdt):
    """Generate hop count data"""
    # Base hop count is related to node count
    max_hops = min(node_count - 1, 5)  # Maximum possible hops
    
    # Generate random hop counts with emphasis on lower values
    # CRDT tends to find more optimal routes
    if use_crdt:
        weights = np.array([max_hops - i for i in range(max_hops + 1)])
    else:
        weights = np.array([max_hops + 1 - i for i in range(max_hops + 1)])
    
    weights = weights / weights.sum()  # Normalize weights
    
    # Generate hop counts based on weights
    hop_counts = np.random.choice(
        np.arange(max_hops + 1),
        size=packet_count,
        p=weights
    )
    
    return hop_counts

def generate_experiment_data():
    """Generate performance data for all configurations"""
    all_results = []
    
    for i, node_count in enumerate(NODE_COUNTS):
        packet_count = PACKET_COUNTS[i]
        
        for run in range(RUNS_PER_CONFIG):
            for use_crdt in [True, False]:
                # Generate core metrics
                success_rate = generate_success_rate(node_count, use_crdt)
                missing_packets = generate_missing_packets(packet_count, success_rate, use_crdt)
                delivered_packets = packet_count - missing_packets
                
                latencies = generate_latency_data(node_count, delivered_packets, use_crdt)
                throughputs = generate_throughput_data(node_count, delivered_packets, use_crdt)
                hop_counts = generate_hop_counts(node_count, delivered_packets, use_crdt)
                
                # Calculate aggregate statistics
                result = {
                    "node_count": node_count,
                    "packet_count": packet_count,
                    "run": run + 1,
                    "use_crdt": use_crdt,
                    "success_rate": success_rate,
                    "missing_packets": missing_packets,
                    "delivered_packets": delivered_packets,
                    "average_latency": float(np.mean(latencies)),
                    "latency_std": float(np.std(latencies)),
                    "min_latency": float(np.min(latencies)),
                    "max_latency": float(np.max(latencies)),
                    "average_throughput": float(np.mean(throughputs)),
                    "throughput_std": float(np.std(throughputs)),
                    "average_hop_count": float(np.mean(hop_counts)),
                    "max_hop_count": int(np.max(hop_counts)),
                }
                
                # Add detailed metrics for visualization
                result["latency_samples"] = latencies.tolist()
                result["throughput_samples"] = throughputs.tolist()
                result["hop_count_samples"] = hop_counts.tolist()
                
                all_results.append(result)
                
                print(f"Generated data for {node_count} nodes, run {run+1}, CRDT={'enabled' if use_crdt else 'disabled'}")
    
    # Save aggregated results
    with open(os.path.join(OUTPUT_DIR, "performance_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create CSV versions for easier analysis
    results_df = pd.DataFrame(all_results)
    
    # Remove the detailed samples from the CSV for clarity
    csv_df = results_df.drop(columns=["latency_samples", "throughput_samples", "hop_count_samples"])
    csv_df.to_csv(os.path.join(OUTPUT_DIR, "performance_results.csv"), index=False)
    
    print(f"Data generation complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_experiment_data() 