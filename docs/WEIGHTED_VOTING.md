# Weighted Voting and Reputation Design for Mesh Offline Payments

## Purpose
Enable local authorization (e.g., issuance top-ups, conflict arbitration, emergency revocations) in offline partitions with resilience to Sybil and collusion, while preserving privacy and auditability.

## Roles
- Issuers/Gateways: mint offline credentials against escrow
- Merchants: accept payments and participate in local decisions
- Agents/Community Validators: provide attestations and monitoring
- Users: transact and may delegate reputation via attestations

## Weights and Identity
- Identity: public keys bound to device/organization with attestations from known anchors
- Weight sources:
  - Stake: locked collateral slashed on fraud
  - Usage: volume and longevity of honest participation (with decay)
  - Attestations: endorsements from reputable entities (web-of-trust)
- Weight decay: exponential decay to prevent indefinitely dominant actors
- Sybil resistance: cap per-anchor endorsements; require heterogeneous attestation origins

## Decision Types and Thresholds
- Emergency Revocation: fast track; threshold by weighted supermajority within locality
- Parameter Updates (caps/gossip): slower path; higher quorum
- Dispute Arbitration: committee randomly sampled weighted by reputation; revealable VRF for audit

## Privacy Considerations
- Aggregate commitments for votes; reveal minimal necessary proofs
- Use one-time voting credentials derived from long-term keys

## Auditability and Slashing
- On reconnection, publish vote transcripts and ZK proofs of proper tally
- Misbehavior (double-voting, invalid votes) yields slashing of stake and reputation decay

## Collusion Resistance
- Randomized committee selection via VRF; require geographic/logical diversity constraints
- Cap influence per community; cross-community veto for high-impact actions

## Integration with Payments
- Merchants can check locally cached quorum artifacts before accepting high-value payments
- Weighted approvals may raise temporary caps for vetted merchants during partitions