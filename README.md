# An Operational Representation of Systems: A Worked Construction in Lattice Gauge Theory

Edward Chalk, Sapientronic — edward.chalk@sapientronic.ai

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23454340.svg)](https://doi.org/10.5281/zenodo.23454340)

## What this is

This repository holds a paper and the code that produces every number in it.

The paper takes a representation of systems in general, the *minimal ordinal systems representation* (MINOS), and carries it
into lattice gauge theory. The representation starts from the smallest account of a system one can give: something drives it,
something constrains it, and how the two cohere is supplied by the system itself. Its one rule of evidence is that only the
*shares* of a state can be read; the frame the state is written in, and any phase it carries, cannot. The paper asks what follows
when that rule is applied to a chain of such systems, and finds that the structures of lattice gauge theory follow. The relation
between two systems' frames is a group element, and the group is classified. The unreadability of frames is a local gauge freedom
and Gauss's law. The cost of carrying a state round a closed path, read as a share and averaged, is a Wilson action, in the adjoint
representation when amplitudes are complex and in the traceless symmetric representation when they are real. The invariant metric
on relations has the Casimir operator as its Laplacian. And if the readable "poles" of the systems are kept as dynamical variables
rather than averaged away, the same construction fixes a definite interaction between neighbouring plaquettes, with no free
coefficient of its own.

The paper's purpose is methodological. It tests whether a representation devised for systems can be stated precisely, applied to an
established mathematical theory without renaming its objects, and used to calculate things that can be checked. It does not derive
Yang–Mills theory from nothing, makes no claim about the continuum limit or the mass gap, and states every assumption it uses.

## What the code demonstrates

`code/minos_yang_mills.py` is one self-contained script. It runs twenty-one numbered checks, each of which tests a specific claim
of the paper against a direct computation, and writes what it finds to `results.txt` (readable) and `results.json` (every number).
The paper's tables are copied from that output. In plain terms, the checks show:

| checks | what is tested | what comes out |
|---|---|---|
| 1–2 | the single element: its relation is a rotation, what is readable of it is quadratic in the amplitudes; two elements in one plane close, in alternating planes they leave a residual | the residual is second order in the mediations |
| 3–5 | the cost of a loop averaged over the element's state, in two readings (amplitudes and shares) | the Wilson action and the adjoint action; the same leading local term at weak coupling with couplings in the ratio 2N/(N+1) |
| 6–7 | the smallest closed loop solved exactly | only whole carriers in the readable vacuum; the gap ratio tends to C_A/C_F = 8/3 at strong coupling and 1 at weak; the held share oscillates at the gap, with the exact spectral weights given |
| 8–9 | the complex branch: the Haar second moment and the information distance between relations | the adjoint action formula to 10⁻¹⁶; the distance is bi-invariant and its Laplacian is the Casimir |
| 10–11, 13 | the real branch: two real elements compose to SO(3) on the real projective plane, and their loss is the spin-two action; the general real formula for N poles | (2/3)(1 − χ₂/5) to the fourth decimal; the traceless symmetric action for N = 3 to 6 |
| 12 | the two invariant measures (Fubini–Study and Fisher–Rao) told apart | average returns 1/3 and 1/2 on the same rotation |
| 14 | composition as restriction of amplitudes | defined everywhere; the shared pole counted in both weights |
| 15–16 | the lattice construction with the poles as transported variables, counted exactly on a triangle and on a four-element graph | every link free; 6 and 8 physical degrees of freedom; the alternative readings give 3 and 6 |
| 17 | whether averaging over the pole gives an invariant sector of the full dynamics | it does not: P V P = 0.8 but ‖(I − P) V P‖ = 0.2667 |
| 18–19 | the two-plaquette interaction induced through a shared pole: its closed-form kernel, and the complete fixed-link Hamiltonian on two squares sharing an element | kernel matched to 10⁻¹⁶; residual against the formula steady in 1/κ² |
| 20–21 | the reduction to the pure gauge theory with the link itself dynamical, by direct diagonalisation of up to 18,876 states in the gauge-invariant sector | the effective operator matches the full spectrum to order 1/κ², across truncations and the four lowest levels |

The checks that carry the paper's argument are 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20 and 21 (paper, §7). Two instrument faults
found while building checks 20 and 21 are kept in the output beside their corrected forms, because the paper's method is that a
miss is recorded, not erased.

## Running it

```
cd code
pip install -r requirements.txt
python minos_yang_mills.py
```

About fifteen minutes on a laptop, most of it check 21. The script is seeded, so the output is reproducible to the printed digits.
Checks 17–21 need scipy and are skipped without it. `requirements.txt` pins the versions the published results were produced with
(Python 3.13.14, numpy 2.2.6, scipy 1.16.3).

## Layout

| | |
|---|---|
| `paper/minos-a-minimal-ordinal-representation-of-systems-worked-construction-in-lattice-gauge-theory.tex` | the paper (LaTeX source) |
| `paper/minos-a-minimal-ordinal-representation-of-systems-worked-construction-in-lattice-gauge-theory.pdf` | the paper, built from the source |
| `code/minos_yang_mills.py` | the script that produces every check |
| `code/results.txt`, `code/results.json` | its output, as used in the paper |
| `code/requirements.txt` | pinned dependencies |
| `CITATION.cff` | citation metadata |

## Licence

The code (everything under `code/`) is released under the MIT licence (`code/LICENSE`). Use, modify and redistribute it freely,
keeping the copyright and licence notice.

The paper (everything under `paper/`) is © 2026 Edward Chalk, all rights reserved. It may be read and shared in its published form.

## Citation

If you use the code or build on the paper, please cite:

> E. Chalk, *An Operational Representation of Systems: A Worked Construction in Lattice Gauge Theory* (2026).
> Zenodo. https://doi.org/10.5281/zenodo.23454340

The DOI resolves to the archived copy of the release on Zenodo; the repository itself is at
https://github.com/pcoz/operational-representation-of-systems. `CITATION.cff` carries the same information for reference managers.
