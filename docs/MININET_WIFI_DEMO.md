# Mininet-WiFi Demo Plan

## Objectives
- Demonstrate real-time offline payment acceptance and receipt gossip over a wireless mesh
- Showcase resilience under partitions, mobility, and delayed reconciliation
- Collect reproducible metrics aligned with Evaluation Plan

## Environment
- Mininet-WiFi (version pinned)
- Python 3.10+
- Dependencies: networkx, matplotlib, scapy (for packet capture), pandas

## Topologies
- Grid 4x4 with 1–3 hop paths
- Random geometric with variable density
- Two-cluster partition bridged intermittently

## Roles
- issuer0: issues offline credentials, publishes CRLs when online
- merchants: m1..mN run merchant daemon
- users: u1..uM run wallet daemon, move along paths

## Scenarios
1. Baseline Honest Traffic
   - Users perform payments; merchants accept; receipts gossip; reconcile later
2. Partition and Merge
   - Cut links between clusters for T minutes; attempt double-spend across partitions
3. Mobility Stress
   - Users move; ensure payments across changing routes
4. Power Loss
   - Kill merchant/user process mid-payment; restart and reconcile

## Metrics Collection
- Per-payment latency, success, hop count
- Receipt propagation time and coverage
- Attack success rate and detection latency
- Bytes/message and CPU proxy for energy

## Scripts
- setup_topology.py: build topology and assign roles
- run_scenario.py: execute scenarios with parameter sweeps
- collect_metrics.py: parse logs and produce CSVs
- plot_results.py: generate figures for slides

## Reproducibility
- Seeded randomness; config files checked in under configs/
- Single command runner to reproduce main figures

## Next Steps
- Implement minimal merchant/wallet daemons with mocked crypto first
- Integrate real crypto primitives and credential logic