# Practical Workarounds in Low-Connectivity Regions (Survey)

## Observed Mechanisms
- Agent Networks: cash-in/cash-out via human agents (e.g., mobile money ecosystems)
- USSD/STK: low-bandwidth interactive sessions; still require live connectivity to operator
- Vouchers/Scratch Cards: prepaid value redeemed later; offline distribution with batch reconciliation
- Airtime-as-Currency: informal exchange of airtime credits as proxy money
- Offline Cards/Terminals: NFC/smartcard with offline counters and merchant risk limits
- Paper/QR Vouchers: one-time codes or signed QR redeemed when connectivity returns
- Community Currencies: local vouchers (e.g., community-issued tokens) with periodic audit

## Lessons for Design
- Bounded Risk: merchants accept offline value only within set limits
- Reconciliation Windows: periodic sync to settle and detect fraud
- Agent Trust Models: leverage local reputation and social collateral
- Simplicity and Robustness: minimal UI flows, tolerance to device loss and SIM swaps
- Multi-Channel Communication: SMS/USSD/NFC/mesh as complementary paths
- Recovery: social recovery and paper backups are practical and expected

## Implications for Proposed System
- Enforce per-device offline caps and credential expiries
- Provide simple receipt mechanisms (printed/QR/NFC) as fallbacks
- Support agent roles with enhanced limits and auditing
- Prioritize low-bandwidth gossip and store-and-forward