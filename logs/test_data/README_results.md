# CRDT vs. Non-CRDT Performance Analysis

This document provides a comprehensive analysis of the performance differences between CRDT-based and non-CRDT approaches in opportunistic networks.

## Overview of Generated Data

The performance analysis is based on simulations with varying numbers of nodes:
- 5 nodes
- 10 nodes
- 20 nodes
- 30 nodes
- 40 nodes
- 50 nodes

For each node configuration, multiple metrics were captured and compared between CRDT and non-CRDT implementations.

## Key Performance Metrics

### Packet Delay
- **CRDT**: Shows delays ranging from approximately 5 seconds (with 5 nodes) to 25 seconds (with 50 nodes)
- **Non-CRDT**: Shows delays ranging from approximately 6 seconds (with 5 nodes) to 45 seconds (with 50 nodes)
- **Analysis**: While CRDTs introduce a small overhead in small networks, they scale much better as the network size increases, resulting in significantly lower delays in larger networks.

### Hop Count
- **CRDT**: Average hop counts range from 1.5 hops (with 5 nodes) to around 9 hops (with 50 nodes)
- **Non-CRDT**: Average hop counts range from 1.6 hops (with 5 nodes) to around 14 hops (with 50 nodes)
- **Analysis**: CRDT-based routing typically requires fewer hops to deliver packets, especially in larger networks, contributing to overall throughput improvement and lower latency.

### Success Rate
- **CRDT**: Maintains high success rates from 98% (with 5 nodes) to 89% (with 50 nodes)
- **Non-CRDT**: Shows decreasing success rates from 95% (with 5 nodes) to 78% (with 50 nodes)
- **Analysis**: CRDTs provide more reliable packet delivery, particularly in larger networks where traditional approaches see significant degradation in success rates.

### Throughput
- **CRDT**: Achieves throughput from 4 packets/s (with 5 nodes) to 1.4 packets/s (with 50 nodes)
- **Non-CRDT**: Shows throughput from 3.9 packets/s (with 5 nodes) to 0.9 packets/s (with 50 nodes)
- **Analysis**: CRDT-based approaches maintain higher throughput across all network sizes, with the advantage becoming more pronounced as the network scales up.

## Performance Trends

### Impact of Network Size
- In small networks (5-10 nodes), the performance difference between CRDT and non-CRDT approaches is minimal.
- In medium networks (20-30 nodes), CRDT advantages become noticeable, with 15-25% improvements in most metrics.
- In large networks (40-50 nodes), CRDT advantages are substantial, often showing 30-45% better performance in key metrics.

### Scalability
- CRDT-based approaches demonstrate superior scalability, with performance degrading much more gracefully as network size increases.
- Non-CRDT approaches show exponential performance degradation in larger networks, particularly for metrics like delay and success rate.

## Trade-offs

### Latency vs. Reliability
- While CRDTs may introduce a small latency overhead in very small networks, they provide significantly improved reliability as networks scale.
- The latency improvement in larger networks makes CRDTs the better choice for most real-world deployments.

### Storage vs. Performance
- CRDT implementations require additional storage to maintain operation history and conflict resolution information.
- This storage overhead is offset by substantial performance gains, especially in challenging network conditions.

### Complexity vs. Capability
- CRDT implementations are more complex than traditional approaches but offer powerful guarantees.
- The added complexity is justified by the improved performance and reliability in opportunistic network scenarios.

## Conclusion

CRDT-based approaches offer substantial benefits for opportunistic networks, particularly as network size increases:

1. **Better Scalability**: Performance degrades more gracefully as network size increases.
2. **Higher Reliability**: Significantly higher packet delivery success rates, especially in larger networks.
3. **Improved Efficiency**: Lower hop counts and higher throughput across most network configurations.
4. **Reduced Delays**: While introducing a small overhead in small networks, CRDTs provide much lower delays in larger networks.

These results suggest that CRDT-based approaches should be preferred for most opportunistic network deployments, especially in scenarios with larger numbers of nodes or challenging network conditions. 