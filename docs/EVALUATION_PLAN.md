# Evaluation Plan

## Goals
- Validate real-time offline payment success under varying connectivity and power conditions
- Measure bounded-loss guarantees and fraud detection latency
- Quantify performance overheads (latency, throughput, energy) on commodity devices

## Metrics
- Payment success rate (%) and median/95p latency (ms)
- Double-spend acceptance rate under attack and detection latency (s)
- Bounded loss vs. configured offline caps (currency units)
- Reconciliation time after connectivity restoration (s)
- Gossip overhead: messages per payment, bytes per node per hour
- Energy consumption: per-transaction and per-hour idle/active
- Storage growth: receipts/CRLs per device per day

## Scenarios
1. Connectivity Duration
   - Short outage: 5–30 minutes
   - Medium outage: 1–6 hours
   - Long outage: 24–168 hours
2. Power Conditions
   - Normal operation
   - User or merchant device power loss during/after payment
3. Network Conditions
   - Stable mesh, 2–6 hops
   - Partitioned mesh merging after T hours
   - Mobility with handoffs and variable link qualities
4. Adversaries
   - Rapid double-spend attempts at k distinct merchants
   - Colluding merchants recycling receipts
   - Sybil nodes inflating reputation/weights

## Experimental Setup (Mininet-WiFi)
- Topologies: grid, random geometric, clustered hotspots
- Node roles: issuers/gateways (few), merchants (edge nodes), users (mobile)
- Emulate partitions: link cuts; delayed CRL propagation
- Traffic mix: background web/VoIP to stress gossip
- Instrumentation: logs, pcap, counters; energy approximated via CPU usage proxy

## Protocol Parameters to Sweep
- Offline cap per device (small, medium, large)
- Credential expiry windows (minutes → days)
- Gossip fanout and interval
- Weighted quorum thresholds and decay

## Statistical Analysis
- Report medians and 95p; bootstrap CIs
- Compare against baselines: naive IOU, payment channels prototype under mesh
- Sensitivity analysis across parameters

## Success Criteria
- ≥99% success for honest payments within target latency under short/medium outages
- Double-spend acceptance ≤ target epsilon; detection within bounded time windows
- Loss exposure ≤ configured caps under all tested conditions