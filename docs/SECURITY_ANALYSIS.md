# Security Analysis Plan

## Assets and Trust Boundaries
- Currency backing (escrows, on-chain funds)
- Offline spend credentials (tokens, signed rights)
- Device keys (user wallet, merchant terminal)
- Receipts and revocation artifacts
- Mesh communication channels and gossip overlays

## Adversary Model
- Malicious user: modifies client to overspend, replay, or clone credentials
- Malicious merchant: attempts to accept invalid payments or extort users
- Colluding group: merchants/users/quorum signers collude to inflate balances or launder double-spends
- Sybil attacker: creates many identities to bias reputation/voting
- Network attacker: partitions, delays, reorders messages; injects or drops gossip
- Key compromise: device theft, malware exfiltration, RNG failure

## Security Goals
- Safety: no accepted payment can violate conservation beyond bounded risk limits
- Liveness: honest payer/payee can complete transactions offline within expected delay
- Double-spend resistance: second acceptance probability is negligible within configured caps
- Accountability: detect and provably attribute fraud on reconnection (fraud proofs)
- Privacy: unlinkability across payments; limited metadata in receipts
- Freshness: replay protection via nonces/one-time credentials and expiry
- Availability: graceful operation under partitions, degraded but functional

## Assumptions
- Cryptographic primitives are secure (signatures, hash functions, commitments)
- Devices maintain secure key storage (software or hardware-backed where available)
- Base chain/issuer eventually available for revocation settlement and slashing
- Time is loosely bounded (credential expiry windows)

## Methodology
1. Threat Modeling
   - STRIDE + attack trees per component (issuance, payment, receipt gossip, reconciliation)
   - Abuse cases: cloned tokens, withheld receipts, quorum collusion
2. Formal Safety/Liveness
   - TLA+ specifications for protocol states: Issued, Spent, Revoked, Reconciled
   - Invariants: conservation within caps, unique-spend property, eventual reconciliation
   - Model partitions and message loss; check liveness under fairness assumptions
3. Cryptographic Protocol Verification
   - ProVerif or Tamarin for authentication, secrecy of blinded tokens, replay resistance
   - Event correspondence: payment implies prior issuance; double-spend yields detectable evidence
4. Compositional Reasoning
   - UC-style ideal functionality for offline credential and receipt gossip; show secure realization under assumptions
5. Economic/Collusion Analysis
   - Define payoffs for collusion; set risk limits/penalties so collusion is irrational beyond small coalitions
   - Use bounded-loss theorem: expected loss ≤ configured caps per epoch

## Artifacts and Outputs
- TLA+ model files and proofs of invariants
- ProVerif/Tamarin models and correspondence queries
- Attack trees and mitigation mapping
- Test vectors and red-team scenarios

## Key Invariants (examples)
- I1: For any credential id, at most one accepted payment exists before expiry unless marked revoked
- I2: Sum(accepted_offline) − Sum(revocations) ≤ GlobalOfflineCap across system
- I3: Any accepted payment implies possession of a valid signature under issuer key at acceptance time

## Test Scenarios
- Rapid double-spend attempts at multi-hop distance
- Partitioned mesh with delayed CRL propagation
- Compromised merchant colluding with users to recycle receipts
- Device key compromise and recovery procedures

## Acceptance Criteria
- All invariants checked in model; counterexamples triaged or design adjusted
- Attacks require breaking crypto or exceeding configured risk bounds
- Privacy properties hold in symbolic model with defined leakage budget