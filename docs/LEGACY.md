# Legacy files: provenance only

These files predate `quhi/`. They are kept on this branch so that git history records what was
audited. They are **provenance only** and must not be merged to `main` as evidence, cited as
results, or imported by code.

- `UHIP_Quantum-main.zip`: the legacy code. 19 of 163 tests fail, and its shipped results are
  penalty artefacts (docs/AUDIT.md F1–F3).
- The 14 root `*.docx` design and maths documents: they contain the errors listed as F4–F7.
- `RESEARCH_EXECUTION_SUMMARY.md`, `IMPLEMENTATION_COMPLETE.md`, `MODULES_SUMMARY.md`,
  `QUANTUM_SOLVERS_IMPLEMENTATION.md`, `SOLVER_INTEGRATION_SUMMARY.md`: summaries of the legacy code.
- `UHIPs.ipynb` and `final qubo uhi jupiter notebook.pdf`: kept as physical intuition only.

`scripts/audit_legacy.py` reads the zip only to reproduce `docs/AUDIT.md`. Nothing in `quhi/`
imports it.

**Suggested handling at merge.** Keep these files out of the tree on `main`, for example by
attaching them to a `legacy/` release asset. The branch history keeps them either way.
`.coverage` is a test artefact and is no longer tracked (it is in `.gitignore`).
