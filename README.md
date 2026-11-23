# Quantum UHI-QUBO

Code and notes for a research framework that formulates urban-heat-island (UHI) green-infrastructure layout as a QUBO (and equivalent Ising) problem, then solves it with classical, quantum-inspired and circuit-based methods.

The Python package itself is not checked out at the repository root. It is bundled in [`UHIP_Quantum-main.zip`](UHIP_Quantum-main.zip), a snapshot of the `UHIP_Quantum` source tree. Design notes are in the Markdown files and Word documents in this directory. [`UHIPs.ipynb`](UHIPs.ipynb) is a short exploratory notebook for green-space placement and cooling on a 2-D grid.

This repository is licensed under the MIT License (see [LICENSE](LICENSE)).

## Getting started

```bash
unzip UHIP_Quantum-main.zip
cd UHIP_Quantum-main
pip install -r ../requirements.txt
```

Examples and tests live inside the zip:

```bash
python examples/spatial_modeling_demo.py
python examples/solver_integration_demo.py
python -m pytest validation/testing/unit_tests
```

Dependencies are listed in [`requirements.txt`](requirements.txt) and include NumPy, SciPy, pandas, PyQUBO, dimod, neal, Qiskit and Qiskit Aer, NetworkX, and plotting and testing tools.

## What is in the source tree

After unzipping, the package is organised as follows (from the modules and the accompanying documentation).

**Problem and QUBO formulation** (`core/problem_modeling`, `core/qubo_formulation`)

- Urban green-layout problem setup (`uhi_green_layout.py`), including grid construction and solution checks.
- Spatial cooling kernels (`cooling_kernels.py`): Gaussian, exponential, power-law and compact kernels, with an anisotropic option for wind.
- Constraint construction (`constraint_builder.py`): resource (area, budget), cardinality, spatial (connectivity, separation) and equity (district) constraints.
- QUBO matrix assembly (`qubo_matrix_builder.py`), combining a physical cooling objective with penalty terms. The notes refer to a penalty-weight rule (Theorem IV.1) that chooses a penalty larger than the range of the physical objective.
- Bidirectional QUBO ↔ Ising conversion (`ising_converter.py`), with a documented equivalence check.

**Spatial modelling** (`core/spatial_modeling`)

- Urban grid, connectivity analysis, and a heat-diffusion simulation used to measure cooling from a candidate layout.

**Solvers** (`core/solvers`, `core/solver_integration`)

- Classical: simulated annealing, tabu search, and a genetic algorithm, behind a common `SolverInterface`.
- Quantum-inspired: a Metropolis sampler, simulated quantum annealing, and a Trotter-expansion routine.
- A shared result type recording the binary solution, energy, runtime and optional convergence data.

**Quantum circuits** (`core/quantum_circuits`)

- QAOA and VQE ansatz construction, gate decomposition, a backend interface, and noise-mitigation helpers, intended for Qiskit / Qiskit Aer.

**Documentation and design notes** (repository root)

- [`MODULES_SUMMARY.md`](MODULES_SUMMARY.md), [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md), [`QUANTUM_SOLVERS_IMPLEMENTATION.md`](QUANTUM_SOLVERS_IMPLEMENTATION.md), [`SOLVER_INTEGRATION_SUMMARY.md`](SOLVER_INTEGRATION_SUMMARY.md), and [`RESEARCH_EXECUTION_SUMMARY.md`](RESEARCH_EXECUTION_SUMMARY.md) describe the modules and a February 2026 analysis run (10×10 grid, 41 candidate sites, comparison of SA, tabu search, GA and simulated quantum annealing).
- The `.docx` and `.pdf` files are the longer design and mathematics notes that those modules cite.

## Layout of this repository

| Path | Role |
|---|---|
| `UHIP_Quantum-main.zip` | Python package, examples, scripts and tests |
| `UHIPs.ipynb` | Exploratory green-space / cooling notebook |
| `requirements.txt` | Python dependencies |
| `*.md` | Module and execution notes |
| `*.docx`, `*.pdf` | Design and mathematics documents |
| `LICENSE` | MIT |

A coverage data file that had been committed by mistake is no longer tracked; `.coverage` is listed in `.gitignore`.
