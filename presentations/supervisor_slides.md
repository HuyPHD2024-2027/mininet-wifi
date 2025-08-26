# Offline Payments over Mesh Networks

## Motivation
- Digital payments fail under connectivity gaps; cash alternatives are risky
- Real-world demand in low-connectivity regions; agent/voucher ecosystems exist

## Vision
- Secure, privacy-preserving, economically sustainable offline payments over wireless mesh
- Real-time acceptance; asynchronous settlement; bounded loss

## Contributions
- Bounded-risk offline credential design with privacy
- Mesh-native receipt/CRL gossip and reconciliation
- Weighted voting for local authorization and safety rails
- Systematic security analysis and formal verification plan
- Mininet-WiFi prototype and evaluation methodology

## Why Not Channels/Rollups?
- Require monitoring, timelocks, global DA; fragile under partitions
- Our model: no watchtowers during payment; losses bounded; local validity

## Security
- Threat models: malicious users/merchants, collusion, Sybil, partitions
- Invariants: conservation within caps, unique-spend, accountability
- Methods: TLA+ safety/liveness; ProVerif/Tamarin crypto; economic bounds

## Evaluation
- Scenarios: outages (minutes→days), partitions/merges, mobility, power loss
- Metrics: success/latency, bounded loss, detection/reconciliation, overhead, energy

## Demo Plan (Mininet-WiFi)
- Topologies: grid, random, partitioned clusters
- Scripts: setup_topology, run_scenario, collect_metrics, plot_results

## Practical Grounding
- Survey of existing workarounds: agents, USSD, vouchers, NFC
- Design lessons: caps, reconciliation windows, recovery, multi-channel comms

## Timeline
- M1: Survey + differentiation drafts
- M2: Security plan + TLA+ skeleton
- M3: Prototype + baseline experiments
- M4: Weighted voting + privacy
- M5: Integrated evaluation + submission prep

## Risks & Mitigations
- Partition fraud → strict caps, expiries, fast CRLs
- Collusion → diversified quorums, slashing, decay
- Proof complexity → staged proofs, simplify where needed
- UX under power loss → recovery flows, NFC/paper backups

## Ask
- Feedback on feasibility, scope focus, and milestone alignment