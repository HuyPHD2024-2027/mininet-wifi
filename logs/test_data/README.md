# CRDT vs. Non-CRDT Performance Comparison

This folder contains scripts and data to analyze the performance differences between using CRDTs (Conflict-Free Replicated Data Types) and traditional approaches in opportunistic networks with varying numbers of nodes.

## Contents

- `generate_data.py`: Generates random simulated performance data for CRDT vs. non-CRDT approaches
- `visualize_data.py`: Creates visualizations comparing performance metrics between approaches
- `analyze_real_data.py`: Analyzes real experiment data and compares with simulations
- `run_all.py`: Runs all scripts in sequence to generate data and visualizations
- `performance_results.json`: Raw performance data (generated) with detailed metrics
- `performance_results.csv`: Summarized performance data in CSV format
- `README_results.md`: Analysis of the findings and performance implications
- `figures/`: Directory containing visualization outputs
- `real_data_figures/`: Directory containing real data visualizations

## How to Run

1. **Generate Data and Visualizations (All Steps)**:
   ```bash
   python run_all.py
   ```
   This runs all the scripts in sequence, generating data, creating visualizations, and analyzing real data.

2. **Individual Steps**:
   ```bash
   # Generate random data only
   python generate_data.py
   
   # Create visualizations only
   python visualize_data.py
   
   # Analyze real data only
   python analyze_real_data.py
   ```

## Metrics Compared

The scripts compare the following performance metrics between CRDT and non-CRDT approaches:

1. **Latency**: Average packet delivery delay in seconds
2. **Throughput**: Average rate of successful packet delivery (packets/second)
3. **Success Rate**: Percentage of successfully delivered packets
4. **Missing Packets**: Number of packets that failed to reach destination
5. **Hop Count**: Average number of hops for packet delivery

## Node Configurations

The comparison includes simulations with varying number of nodes:
- 3 nodes
- 5 nodes
- 7 nodes
- 10 nodes
- 15 nodes

## Running Your Own Experiments

To run your own experiments with different numbers of nodes, you can modify the `generate_data.py` script. Look for the following variables at the top of the file:

```python
# Configuration parameters
NODE_COUNTS = [3, 5, 7, 10, 15]  # Different numbers of nodes to simulate
PACKET_COUNTS = [100, 150, 200, 250, 300]  # Corresponding packets per node count
RUNS_PER_CONFIG = 5  # Number of runs per configuration
```

You can adjust these parameters to test different network configurations:

1. **NODE_COUNTS**: Change this list to test different numbers of nodes
2. **PACKET_COUNTS**: Adjust the number of packets for each node count configuration
3. **RUNS_PER_CONFIG**: Increase for more statistically significant results

Additionally, you can modify the performance factors to test different hypotheses about CRDT behavior:

```python
# Performance factors (how much CRDT improves/degrades performance)
# Values above 1.0 mean CRDT performs worse, below 1.0 means CRDT performs better
LATENCY_FACTOR = 1.15  # CRDT slightly increases latency
THROUGHPUT_FACTOR = 0.85  # CRDT improves throughput
SUCCESS_RATE_FACTOR = 0.7  # CRDT significantly improves success rate
MISSING_PACKETS_FACTOR = 0.6  # CRDT reduces missing packets
```

## Running Real Mininet-WiFi Experiments

To run actual experiments on Mininet-WiFi with different node counts:

1. Create a new topology script in your Mininet-WiFi project:
   ```python
   from mininet.log import setLogLevel
   from mn_wifi.cli import CLI
   from mn_wifi.net import Mininet_wifi
   from mn_wifi.opportunisticNode import OpportunisticNode
   from mn_wifi.opportunisticLink import opportunisticLink
  
   def topology(node_count=5):
       "Create a network with specified number of nodes"
       net = Mininet_wifi(station=OpportunisticNode, link=opportunisticLink)
      
       print("*** Creating nodes")
       # Create the desired number of nodes
       nodes = []
       for i in range(node_count):
           node = net.addStation(f'node{i+1}', position=f'{i*50},50,0')
           nodes.append(node)
          
       # Configure CRDT
       print("*** Configuring CRDT")
       for node in nodes:
           # Configure CRDT parameters here
           pass
          
       # Rest of your topology setup
       # ...
      
       return net
      
   if __name__ == '__main__':
       setLogLevel('info')
       # Adjust node_count as needed
       net = topology(node_count=7)  # Test with 7 nodes
       # ...
   ```

2. Run your experiment with different node counts and collect data for each run.

3. Compare the results to understand how the number of nodes affects performance in real-world conditions.

## Files Generated

The visualization script generates the following files in the `figures/` directory:

- `latency_comparison.png`: Compares average latency between approaches
- `throughput_comparison.png`: Compares throughput between approaches
- `success_rate_comparison.png`: Compares packet delivery success rate
- `missing_packets_comparison.png`: Compares percentage of missing packets
- `hop_count_comparison.png`: Compares average hop count for delivery
- `radar_comparison.png`: Radar chart comparing all metrics (10-node configuration)
- `improvement_percentage.png`: Shows percentage improvement from using CRDTs
- `performance_summary.csv`: Summary table of all metrics in CSV
- `performance_summary.html`: Summary table in HTML format for easy viewing 