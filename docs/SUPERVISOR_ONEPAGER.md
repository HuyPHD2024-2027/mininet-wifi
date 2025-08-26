# One-Pager: Offline Payments over Mesh Networks

## Problem
Digital payments fail during connectivity outages; current blockchain scaling assumes online monitoring and global availability.

## Proposed Solution
A bounded-risk, privacy-preserving offline payment system over wireless mesh that enables real-time acceptance with asynchronous reconciliation to a base chain or issuer.

## Key Ideas
- Offline credentials with per-device caps and expiries
- Mesh-native receipt/CRL gossip; no watchtowers during payment
- Weighted voting for local authorization and emergency actions
- Accountability via fraud proofs and slashing

## Why Different
Payment channels/rollups require connectivity and monitoring; our approach tolerates prolonged partitions with bounded loss.

## Security Plan
- Threat models (malicious users/merchants, collusion, Sybil, partitions)
- Formal invariants in TLA+; crypto verification in ProVerif/Tamarin
- Economic analysis to set safe caps and penalties

## Evaluation Plan
- Mininet-WiFi prototype; scenarios: outages, partitions/merges, mobility, power loss
- Metrics: success/latency, bounded loss, detection/reconciliation, overhead, energy

## Milestones
- M1: Survey + differentiation drafts
- M2: Security plan + TLA+ skeleton
- M3: Prototype + baseline experiments
- M4: Weighted voting prototype
- M5: Integrated evaluation + submission prep

## Risks & Mitigations
- Partition fraud → caps/expiries/CRLs
- Collusion → diverse quorums/slashing/decay
- Proof complexity → staged proofs

## Ask
Feedback on scope and milestone alignment; green-light for prototype build.