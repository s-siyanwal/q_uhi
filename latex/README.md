# LaTeX sources for the `quhi` papers

Three PDFs are built from the same facts:

| output | source | class | pages |
|---|---|---|---|
| `results/papers/ieee_quhi.pdf` | `ieee/main.tex` | `IEEEtran` (conference, two-column) | 8 |
| `results/papers/springer_quhi.pdf` | `springer/main.tex` | Springer **LNCS** (`llncs`, `splncs04.bst`) | 11 |
| `results/papers/quhi_full_report.pdf` | `report/main.tex` + `report/appendix_*.tex` | `report` (Times via `mathptmx`) | 40 |

## Build

```bash
make -C latex all      # regenerate tables from results/*.csv, build ieee, report, springer, copy PDFs
make -C latex ieee     # one document (also: springer, report, tables, copy, clean)
```

Each target runs `latexmk -pdf -interaction=nonstopmode -halt-on-error`. The build was tested with
TeX Live 2023 (Ubuntu packages `texlive-latex-base texlive-latex-recommended
texlive-fonts-recommended texlive-publishers latexmk`). `IEEEtran.cls` and `llncs.cls` come from
TeX Live (`texlive-publishers`), so neither class is vendored here. No `siunitx`, `enumitem`,
`tcolorbox` or `lmodern` is needed.

## Layout

- `shared/`: `macros.tex` (notation), `abstract.tex` (176 words), `acronyms.tex`, `acknowledgements.tex`.
- `content/01_intro.tex … 11_limitations.tex`: section bodies. The LNCS paper uses all of them.
  The report reuses them where they fit. The IEEE paper has its own condensed prose, so the two
  papers are not a class swap of one file.
- `tables/`: `*.tex` in the main text and `appendix/*_full.tex` in the report.
  - Most are **generated** from the committed CSVs by `tools/make_tables.py`: E1, E2, E3, E5,
    E8b, E9, E10, E12 and E12b, plus every appendix table.
  - The rest are **typed** from `docs/RESULTS.md`, `docs/AUDIT.md` or `results/E13_tile/e13.md`:
    `audit.tex`, `e4.tex`, `e6.tex`, `e7.tex`, `e8.tex`, `e11.tex` and `e13.tex`.
  - No number is computed that is not already in those files.
- `figs/`: copies of existing PNGs only. Nothing was regenerated.

  | figure | source |
  |---|---|
  | `p_opt_vs_p.png` | `results/E8b_depth/` |
  | `compare_final.png`, `compare_final_B.png` | `results/ANIM/` |
  | `e12_benefit_vs_time.png` | `results/E12_comparable/` |
  | `e13_city.png`, `e13_plans.png` | `results/E13_tile/` |
  | `e2_fidelity.png` | `results/E2_fidelity/` |
  | `e3_benefit_mix.png` | `results/E3_scaling/` |
  | `e4_penalty_sweep.png` | `results/E4_penalty/` |
  | `e6_gap_vs_penalty.png` | `results/E6_quantum/` |
  | `e7_transition.png` | `results/E7_showcase/` |

  All requested figure paths existed, so no figure was dropped. The GIFs are not embedded; the
  report links them by GitHub blob URL.
- `bib/refs.bib`: only the works that are actually cited. Each entry comes from
  `docs/BACKGROUND.md`, except HiGHS (Huangfu & Hall 2018). The only DOI given is the one in
  `docs/BACKGROUND.md` (Zhou et al. 2023). No other DOIs were added.

## Rules the sources follow

- Every number appears in `docs/{PAPER,RESULTS,AUDIT,NEXT}.md`, `results/*/summary.csv` or
  `results/*/e*.md`.
- The legacy zip energies are not cited; they are F1–F3 artefacts.
- There are no hardware, D-Wave or Leap numbers, and no quantum-advantage claim.
- Methods are frozen, and no experiment was rerun to build these documents.
