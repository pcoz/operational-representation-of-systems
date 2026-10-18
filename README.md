# An Operational Representation of Systems: A Worked Construction in Lattice Gauge Theory

Edward Chalk, Sapientronic — edward.chalk@sapientronic.ai

A representation of systems in general — the minimal ordinal systems representation, MINOS — is carried into lattice gauge theory
and tested there. From one rule of evidence, that only the shares of a state can be read and not its frame or its phase, with its
assumptions stated and counted, the paper produces the gauge group and its classification, the local gauge freedom and Gauss's law, the invariant measure,
the Wilson character actions (adjoint for complex amplitudes, traceless symmetric for real), and the Casimir kinetic term; and, with
the readable poles kept as dynamical variables, a definite two-plaquette interaction whose form the model fixes. Every number in the
paper is produced by the accompanying code.

| | |
|---|---|
| `paper/main.tex` | the paper (LaTeX; `paper/main.pdf` is the built copy) |
| `code/minos_yang_mills.py` | the one script that produces every check (1–21); `results.txt` and `results.json` are its output |
| `code/requirements.txt` | the versions the results were produced with (Python 3.13.14, numpy 2.2.6, scipy 1.16.3) |

## Running the checks

```
cd code
pip install -r requirements.txt
python minos_yang_mills.py
```

About fifteen minutes; checks 17–21 need scipy and are skipped without it. The script is seeded and writes `results.txt` and
`results.json` beside itself.

## Licence

The code (everything under `code/`) is released under the MIT licence (`code/LICENSE`): use, modify and redistribute it freely,
keeping the copyright and licence notice.

The paper (everything under `paper/`) is © 2026 Edward Chalk, all rights reserved. It may be read and shared in its published form;
please do not modify or redistribute altered versions.

## Citation

If you use the code or build on the paper, a citation is requested:

> E. Chalk, *An Operational Representation of Systems: A Worked Construction in Lattice Gauge Theory* (2026).

A `CITATION.cff` is provided for reference managers.
