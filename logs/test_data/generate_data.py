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

# Configuration parameters as requested by user
NODE_COUNTS = [5, 10, 20, 30, 40, 50]  # Different numbers of nodes to simulate
PACKET_COUNTS = [100, 150, 200, 250, 300, 350]  # Corresponding packets per node count
RUNS_PER_CONFIG = 5  # Number of runs per configuration

# Custom performance parameters for the requested metrics
# These are calibrated to match the user's requirements

def generate_latency_data(node_count, packet_count, use_crdt):
    """Generate latency data with specific patterns"""
    # For CRDT: 3-8s linear increase
    # For non-CRDT: 3.4-12.2s with exponential increase at higher node counts
    base_latency = 3.0 if use_crdt else 3.4
    
    if use_crdt:
        # Linear increase for CRDT
        max_latency = 8.0
        # Calculate where we are in the range from min to max nodes
        node_factor = (node_count - NODE_COUNTS[0]) / (NODE_COUNTS[-1] - NODE_COUNTS[0])
        # Linear mapping to latency range
        avg_latency = base_latency + node_factor * (max_latency - base_latency)
    else:
        # Exponential increase for non-CRDT to make it worse at higher node counts
        max_latency = 12.2
        # Use a non-linear curve that gets steeper at higher node counts
        if node_count <= 30:
            # More gradual increase up to 30 nodes
            node_factor = (node_count - NODE_COUNTS[0]) / (30 - NODE_COUNTS[0])
            avg_latency = base_latency + node_factor * (7.5 - base_latency)
        else:
            # Steeper increase after 30 nodes
            node_factor = (node_count - 30) / (NODE_COUNTS[-1] - 30)
            avg_latency = 7.5 + node_factor * (max_latency - 7.5) * 1.5  # 1.5x steeper curve
    
    # Generate a distribution around the average latency
    variance = avg_latency * 0.15  # 15% variance
    latencies = np.random.normal(loc=avg_latency, scale=variance, size=packet_count)
    
    # Ensure no negative values
    return np.maximum(latencies, 1.0)  # Ensure latency is at least 1 second

def generate_hop_counts(node_count, packet_count, use_crdt):
    """Generate hop count data with specific patterns"""
    # For CRDT: Always less hop count than non-CRDT at every node count
    base_hop_count = 1.5 if use_crdt else 1.8
    
    if use_crdt:
        # CRDT hop count should be lower than non-CRDT for all node counts
        # and increase more gradually
        max_hop_count = 8.0
        node_factor = (node_count - NODE_COUNTS[0]) / (NODE_COUNTS[-1] - NODE_COUNTS[0])
        avg_hop_count = base_hop_count + node_factor * (max_hop_count - base_hop_count)
    else:
        # Non-CRDT will always have more hops
        max_hop_count = 12.0
        node_factor = (node_count - NODE_COUNTS[0]) / (NODE_COUNTS[-1] - NODE_COUNTS[0])
        # Ensure non-CRDT hop count is always higher than CRDT at the same node count
        # by adding a constant factor
        avg_hop_count = base_hop_count + node_factor * (max_hop_count - base_hop_count) + 1.0
    
    # Calculate a reasonable max for hop counts
    max_hops = min(int(avg_hop_count * 2), node_count)
    
    # Generate random hop counts using a Poisson distribution (good for count data)
    hop_counts = np.random.poisson(lam=avg_hop_count, size=packet_count)
    
    # Ensure hops are within reasonable range
    hop_counts = np.minimum(hop_counts, max_hops)
    hop_counts = np.maximum(hop_counts, 1)  # At least 1 hop
    
    return hop_counts

def generate_success_rate(node_count, use_crdt):
    """Generate success rate data with specific patterns"""
    # For CRDT: 98%-89% with gradual decrease
    # For non-CRDT: 95%-78% with drastic drop after 40 nodes
    max_success_rate = 0.98 if use_crdt else 0.95
    min_success_rate = 0.89 if use_crdt else 0.78
    
    if use_crdt:
        # Gradual decrease for CRDT
        node_factor = (node_count - NODE_COUNTS[0]) / (NODE_COUNTS[-1] - NODE_COUNTS[0])
        success_rate = max_success_rate - node_factor * (max_success_rate - min_success_rate)
    else:
        # For non-CRDT, make the success rate drop drastically after 40 nodes
        if node_count <= 30:
            # Gradual decrease up to 30 nodes
            node_factor = (node_count - NODE_COUNTS[0]) / (30 - NODE_COUNTS[0])
            success_rate = max_success_rate - node_factor * (max_success_rate - 0.88)
        elif node_count <= 40:
            # Moderate decrease from 30 to 40 nodes
            node_factor = (node_count - 30) / 10
            success_rate = 0.88 - node_factor * 0.05
        else:
            # Steep decrease after 40 nodes
            node_factor = (node_count - 40) / 10
            success_rate = 0.83 - node_factor * (0.83 - min_success_rate)
    
    # Add some randomness, but less at the extremes
    variation = 0.01 * (1 - node_factor)
    success_rate += random.uniform(-variation, variation)
    
    # Ensure the result is between 0 and 1
    return max(0.01, min(0.99, success_rate))

def generate_throughput_data(node_count, packet_count, use_crdt):
    """Generate throughput data with specific patterns"""
    # For CRDT: Better throughput than non-CRDT at every node count
    base_throughput = 4.2 if use_crdt else 3.8
    
    if use_crdt:
        # CRDT throughput decreases more gradually
        min_throughput = 1.8
        node_factor = (node_count - NODE_COUNTS[0]) / (NODE_COUNTS[-1] - NODE_COUNTS[0])
        avg_throughput = base_throughput - node_factor * (base_throughput - min_throughput)
    else:
        # Non-CRDT throughput will always be lower than CRDT at the same node count
        min_throughput = 1.2
        node_factor = (node_count - NODE_COUNTS[0]) / (NODE_COUNTS[-1] - NODE_COUNTS[0])
        # Ensure non-CRDT throughput is always lower than CRDT at the same node count
        avg_throughput = base_throughput - node_factor * (base_throughput - min_throughput) - 0.4
    
    # Add some variance
    variance = avg_throughput * 0.1  # 10% variance
    
    # Generate throughput measurements
    throughputs = np.random.normal(loc=avg_throughput, scale=variance, size=min(30, packet_count // 3))
    
    # Ensure no negative values
    return np.maximum(throughputs, 0.1)  # Ensure throughput is at least 0.1

def generate_missing_packets(packet_count, success_rate, use_crdt):
    """Calculate number of missing packets based on success rate"""
    expected_missing = int(packet_count * (1 - success_rate))
    
    # Add some randomness
    variance = int(expected_missing * 0.1)
    missing_packets = max(0, expected_missing + random.randint(-variance, variance))
    
    return missing_packets

def generate_experiment_data():
    """Generate performance data for all configurations"""
    all_results = []
    
    for i, node_count in enumerate(NODE_COUNTS):
        packet_count = PACKET_COUNTS[i % len(PACKET_COUNTS)]
        
        for run in range(RUNS_PER_CONFIG):
            for use_crdt in [True, False]:
                # Generate core metrics
                success_rate = generate_success_rate(node_count, use_crdt)
                missing_packets = generate_missing_packets(packet_count, success_rate, use_crdt)
                delivered_packets = packet_count - missing_packets
                
                latencies = generate_latency_data(node_count, packet_count, use_crdt)
                throughputs = generate_throughput_data(node_count, packet_count, use_crdt)
                hop_counts = generate_hop_counts(node_count, packet_count, use_crdt)
                
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