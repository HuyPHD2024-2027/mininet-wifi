# Offline Payments over Mesh Networks: Consolidated Research Plan

## Purpose
Translate reviewer feedback into a concrete, feasible, and defensible research and engineering plan for a deployable, privacy-preserving, and economically sustainable offline payment system over wireless mesh networks, with rigorous security analysis and clear differentiation from existing blockchain scaling techniques.

## Objectives (tied to feedback)
- Ground proposal in real-world practices in low-connectivity regions by surveying practical workarounds and extracting design lessons.
- Provide a systematic security analysis with explicit threat models and proof methodology suitable for top-tier venues.
- Clearly differentiate the proposed approach from payment channels (e.g., Lightning), rollups, and other scaling techniques under offline constraints.
- Deliver a realistic evaluation plan focusing on connectivity durations, power outages, partitioning, and adversarial behavior, not just wireless specifics.
- Prototype and evaluate with mininet-wifi to emulate mesh, partitioning, and mobility; quantify performance, risk, and safety bounds.
- Explore and specify a weighted voting/reputation mechanism for local consensus or risk authorization in offline settings.

## Proposed Architecture (high-level)
- Asset representation: risk-limited offline spend credentials (escrow-backed) represented as cryptographic tokens or signed spending rights.
- Local validation: merchant devices verify token authenticity using cached public parameters and revocation lists; double-spend resistance via local witness receipts and bounded risk limits.
- Mesh propagation: payment receipts and revocation updates disseminate via mesh gossip; asynchronous settlement to a base chain or authority when connectivity returns.
- Safety rails: per-device offline spend caps, time-locked credentials, and revocable keys to bound fraud exposure.
- Privacy: unlinkability across payments via one-time keys and blinded issuance; local receipts reveal minimal metadata.
- Accountability: fraud proofs upon reconnection enable slashing/penalties of issuers or colluding quorums.

## Work Packages and Deliverables
1) Practical Workarounds Survey (Docs)
- Collect and synthesize mechanisms used in low-connectivity regions (agents, vouchers, NFC cards, SMS/USSD, community currencies).
- Extract requirements: risk limits, reconciliation windows, agent networks, simple UX, recovery.
- Deliverable: PRACTICAL_WORKAROUNDS.md with lessons mapped to design decisions.

2) Systematic Security Analysis
- Assets, adversaries, assumptions; formal properties (safety, liveness, confidentiality, privacy, bounded loss).
- Methodology: TLA+ for protocol safety/liveness; Tamarin/ProVerif for crypto flows; game-theoretic collusion analysis; UC-style abstraction for composition.
- Deliverables: SECURITY_ANALYSIS.md (plan), executable models and proof artifacts (later milestones).

3) Differentiation from Existing Scaling Approaches
- Analysis of payment channels, watchtowers, rollups/validity proofs, sidechains under prolonged disconnection and partitions.
- Show why timelocks, channel monitoring, and DA assumptions fail in offline/mesh settings; position our bounded-risk, asynchronous model.
- Deliverable: DIFFERENTIATION.md with comparison matrix.

4) Evaluation Plan and Metrics
- Metrics: success rate, latency, bounded loss vs. limits, detection latency, reconciliation time, throughput/overhead, energy impact, storage growth.
- Scenarios: outage durations (minutes→days), partitions, mobility, power loss, adversarial behaviors (double-spend, collusion, Sybil).
- Deliverable: EVALUATION_PLAN.md; experiment scripts later.

5) Weighted Voting/Reputation Design
- Define quorum formation, weights (stake/usage/reputation), decay, Sybil resistance (web-of-trust/attestations), and slashing.
- Privacy-preserving aggregation and auditability; resistance to collusion.
- Deliverable: WEIGHTED_VOTING.md.

6) Mininet-WiFi Prototype and Experiments
- Implement wallet/merchant daemons with offline credential verification and receipt gossip.
- Emulate topologies (grid, random, multi-hop), partitions, mobility, and varying link qualities; collect metrics.
- Deliverable: MININET_WIFI_DEMO.md and runnable scripts.

7) Supervisor Presentation Package
- One-page overview and milestone timeline for quick review.
- Slide deck covering motivation, design, security, differentiation, evaluation, demo plan, risks.
- Deliverables: SUPERVISOR_ONEPAGER.md, presentations/supervisor_slides.md, README.md (how to present).

## Milestones (indicative)
- M1 (Month 1-2): Survey + Differentiation drafts complete.
- M2 (Month 2-3): Security analysis plan; initial TLA+ skeletons.
- M3 (Month 3-4): Mininet-wifi prototype scaffolding; baseline experiments.
- M4 (Month 4-5): Weighted voting prototype; privacy/composability notes.
- M5 (Month 5-6): Integrated evaluation run; supervisor review; prepare venue submission outline.

## Risks and Mitigations
- Risk: Unbounded fraud in partitions. Mitigation: strict offline spend caps, short-lived credentials, rapid CRL gossip.
- Risk: Colluding local quorums. Mitigation: diversified quorum selection, stake/reputation weighting with decay and slashing.
- Risk: Complexity of formal proofs. Mitigation: stage proofs (safety first), use established tools, simplify protocol where needed.
- Risk: UX under power loss. Mitigation: recovery flows, paper/NFC backups, social recovery.