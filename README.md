# LPG-Semantic Communication Framework
## A formal extension of Shannon's classical inofrmation
## theory into semantic communication

**Author:** Dada Theophilus Olusegun
**Institution:** Landmark University, Nigeria
**Department:** Electrical and Electronics Engineering  
**Status:** PhD research — internal validation complete real-data validation in progress

---

## What This Is
A rigorous mathematical framework — Logarithmic Probability 
Geometry (LPG) — that formally extends Shannon's 1948 
information theory across the boundary he deliberately left 
open- the transmission of meaning rather than symbols.

Shannon's classical capacity formula is proved to be the 
exact zero-anchor limit of this framework, confirmed 
empirically at SBE = 1.00. The semantic extension delivers:

- 2.04× semantic throughput over Shannon at full anchor depth
- 50% transmit power reduction for equivalent reliability
- 8 dB SNR gain vs classical 5G NR at URLLC threshold
- Adaptive feedback convergence in median 8 rounds
- A new cryptographic primitive defined in meaning space

---

## Repository Contents

| File/Folder | What it contains |
|---|---|
| README.md | This file |
| mechanism_narrative.docx |prose explanation |
| verbal_pipeline.docx | Exhaustive step-by-step verbal account |
| algorithm.docx and pdf| Formal algorithmic specification |
| flowchart.png | Closed-loop pipeline diagram |
| lpg_core.py | Core 1182-line Python implementation |
| simulations | S1–S12 simulation scripts and reports |
| live Demo | S3 and S5 selected to compare LPG AND DeepSC; to demonstrate 6G URLLC at lower SNR|
| results/ | Quantified simulation outputs |
| Summary results of the 12 simulations

---

## Key Claims and Their Confirmation Status

|---|---|---|
| Shannon recovery at ADI=0 | Confirmed S1 | SBE = 1.00 exactly |
| Semantic capacity exceeds Shannon | Confirmed S1 | SBE = 2.04 at ADI≈1 |
| LPG outperforms DeepSC | Confirmed S2 | +31 pts CAR at −6 dB |
| Anchor super-recovery | Confirmed S3 | ARC = 1.17 |
| Cauchy-Riemann conditions computable | Confirmed S4 | CRCS = 0.91 |
| 6G URLLC at lower SNR | Confirmed S5 | −8 dB vs 5G NR |
| Semantic security boundary | Confirmed S6 | cos θ ≈ 0.75 |
| ι irreducibility | Confirmed S8+S10 | δᵢ > 0 always |
| Adaptive feedback convergence | Confirmed S11 | FEG = 1.52 |

---

## Current Status

Internal validation: complete across 12 simulations  
The data used for all the simulations are synthetic
Real-data validation: is the next step, experimental designs ready  
Publication target: IEEE Transactions on Communications

---

## Contact

Dada Theophilus Olusegun  
theodada@gmail.com, dada.theophilus@lmu.edu.ng