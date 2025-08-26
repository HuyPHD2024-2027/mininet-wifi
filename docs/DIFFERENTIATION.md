# Differentiation from Existing Blockchain Scaling Techniques

## Context
Many scaling designs assume frequent connectivity, timely monitoring, and global data availability. Prolonged offline operation and mesh partitions violate these assumptions.

## Payment Channels (e.g., Lightning)
- Requirements: timely channel state updates, monitoring by peers or watchtowers; HTLC timelocks assume bounded delay
- Issues offline: sender/receiver or intermediaries offline break route; missed monitoring risks theft; rebalancing needs connectivity
- Mesh impact: multi-hop paths unstable; timelocks expire under partitions; watchtowers unavailable

## Rollups (Optimistic/Validity)
- Requirements: data availability, periodic posting of batches/proofs
- Issues offline: users can transact locally only if full DA is accessible; dispute windows require online participation
- Mesh impact: DA dissemination is challenging; latency to L1 defeats real-time offline settlement

## Sidechains/Plasma/State Channels Hybrids
- Similar reliance on connectivity, exits, and monitoring; exit games not suitable for prolonged disconnection

## Proposed Approach (Key Differences)
- Bounded-Risk Offline Credentials: spendable without contacting counterparties; losses capped by per-device limits and expiry
- Asynchronous Reconciliation: receipts and revocations settle later; no reliance on timelocks/watchtowers during payment
- Local Validity: merchants verify cryptographic authenticity and freshness from cached parameters and CRLs
- Mesh-Native Gossip: receipts/CRLs propagate opportunistically; no global synchronous consensus required at payment time
- Privacy-Preserving Issuance/Spend: use blinded issuance and one-time keys for unlinkability
- Accountability with Slashing: fraud proofs on reconnection penalize issuers/quorums that misbehave

## When Channels/Rollups Still Help
- Use channels for replenishing escrow when connectivity exists
- Use rollups for batched settlement and fraud-proof arbitration post-facto

## Comparison Matrix (abridged)
- Real-time offline payments: Channels ✗, Rollups ✗, Proposed ✓ (bounded risk)
- Needs watchtowers during payment: Channels ✓, Proposed ✗
- Double-spend during partition: Channels high risk if unmonitored; Proposed bounded by caps and detectable
- Privacy offline: Channels limited; Proposed designed for unlinkability