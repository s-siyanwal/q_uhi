# UHI-QUBO Framework - Core Modules Summary

## Overview
This document summarizes the 5 core modules created for the UHI-QUBO research framework.

## Created Modules

### 1. `core/qubo_formulation/qubo_matrix_builder.py`
**Purpose**: Main QUBO matrix construction class

**Key Features**:
- Combines physical UHI cooling objectives with constraint penalties
- Supports linear and quadratic term addition
- Implements Theorem IV.1 for penalty weight calculation
- Provides solution evaluation and decomposition
- Includes verification and validation methods

**Key Classes**:
- `QUBOBuildConfig`: Configuration dataclass
- `QUBOMatrixBuilder`: Main builder class with methods:
  - `add_linear_terms()`: Add linear coefficients (diagonal)
  - `add_quadratic_terms()`: Add interaction terms
  - `add_physical_objective()`: Add UHI cooling model
  - `add_equality_constraint()`: Add equality constraints
  - `add_inequality_constraint()`: Add inequality constraints  
  - `add_area_constraint()`: Specific for area limits
  - `add_budget_constraint()`: Specific for budget limits
  - `build()`: Finalize and return complete Q matrix
  - `evaluate_solution()`: Decompose energy for a solution

**Mathematical Basis**:
- QUBO form: `f(x) = x^T Q x` where `Q = Q_physical + Σ P_i * Q_constraint_i`
- Penalty weight: `P > range(f_physical)` ensures exact feasibility

**Example Usage**:
```python
config = QUBOBuildConfig(n_variables=10, auto_penalty_weight=True)
builder = QUBOMatrixBuilder(config)
builder.add_physical_objective(alpha, beta)
builder.add_area_constraint(areas, area_max)
Q = builder.build()
```

---

### 2. `core/qubo_formulation/ising_converter.py`
**Purpose**: Bidirectional conversion between QUBO and Ising formulations

**Key Features**:
- Exact QUBO ↔ Ising transformation
- Supports binary ↔ spin variable conversion  
- Verification of equivalence
- Coupling statistics and analysis
- D-Wave Ocean SDK compatible output

**Key Classes**:
- `IsingModel`: Dataclass for Ising Hamiltonian (h, J, offset)
- `IsingConverter`: Static methods for conversion

**Mathematical Transformation**:
- Variable mapping: `x_i = (1 + s_i)/2` where `x ∈ {0,1}`, `s ∈ {-1,+1}`
- QUBO: `E(x) = x^T Q x`
- Ising: `H(s) = -Σ h_i s_i - Σ_{i<j} J_ij s_i s_j + offset`

**Formulas**:
Given symmetric QUBO matrix Q:
```
h_i = -0.5 * Σ_j Q_ij
J_ij = -0.25 * (Q_ij + Q_ji) for i < j  
offset = 0.25 * Σ_i Σ_j Q_ij
```

**Example Usage**:
```python
Q = np.array([[1, 2], [2, 3]])
ising = IsingConverter.qubo_to_ising(Q)
x = np.array([0, 1])
s = IsingConverter.binary_to_spin(x)
is_equiv, msg = IsingConverter.verify_equivalence(Q, ising)
```

---

### 3. `core/problem_modeling/cooling_kernels.py`  
**Purpose**: Spatial cooling kernels for Urban Heat Island modeling

**Key Features**:
- Multiple kernel types (Gaussian, Exponential, Power Law, Compact Support)
- Anisotropic kernels for wind effects
- Efficient kernel matrix computation
- Parameter estimation from data
- QUBO coefficient generation

**Key Classes**:
- `CoolingKernel`: Base class
- `GaussianCoolingKernel`: Diffusion-dominated cooling
  - Formula: `K(r₁, r₂) = A * exp(-||r₁ - r₂||² / (2σ²))`
- `ExponentialCoolingKernel`: Advection-diffusion balance
  - Formula: `K(r₁, r₂) = A * exp(-||r₁ - r₂|| / λ)`
- `PowerLawCoolingKernel`: Long-range interactions
  - Formula: `K(r₁, r₂) = A / (1 + ||r₁ - r₂||/λ)^p`
- `CompactSupportKernel`: Strictly local effects
  - Formula: Wendland C2 kernel with cutoff radius
- `AnisotropicGaussianKernel`: Directional preferences

**Utility Functions**:
- `compute_qubo_cooling_coefficients()`: Generate α and β from kernel
- `estimate_kernel_parameters_from_data()`: Fit kernel to measurements

**Physical Interpretation**:
- `σ` (length scale): Range of cooling influence  
- `A` (amplitude): Maximum cooling strength
- Kernel values represent interaction strength between green spaces

**Example Usage**:
```python
kernel = GaussianCoolingKernel(sigma=100.0, amplitude=-2.0)
positions = np.array([[0,0], [100,0], [0,100]])  # Cell centers
alpha, beta = compute_qubo_cooling_coefficients(positions, kernel)
```

---

### 4. `core/problem_modeling/uhi_green_layout.py`
**Purpose**: Main UHI green space layout problem class

**Key Features**:
- Complete problem formulation and setup
- Grid generation (uniform or from land use map)
- Cooling model integration
- QUBO matrix construction
- Solution validation and statistics
- Export/import capabilities

**Key Classes**:
- `GridCell`: Represents a single urban cell
- `UHIProblemConfig`: Problem configuration
- `UHIGreenLayoutProblem`: Main problem class

**Problem Formulation**:
```
minimize    ΔI_UHI(x) = Σ α_i x_i + Σ β_ij x_i x_j
subject to  Σ area_i x_i ≤ A_max         (area constraint)
            Σ cost_i x_i ≤ B_max         (budget constraint)
            x_i ∈ {0, 1}                (binary decisions)
```

**Workflow**:
1. `__init__(config)`: Initialize with configuration
2. `setup_uniform_grid()` or `setup_from_land_use_map()`: Define cells
3. `setup_cooling_model(kernel)`: Configure physics
4. `build_qubo()`: Construct QUBO matrix with constraints
5. Solve with your preferred solver
6. `set_solution(x)`: Set and validate solution
7. `get_solution_statistics()`: Analyze results

**Example Usage**:
```python
config = UHIProblemConfig(
    grid_rows=10, grid_cols=10,
    cooling_kernel_type='gaussian',
    area_budget_fraction=0.3
)
problem = UHIGreenLayoutProblem(config)
problem.setup_uniform_grid(candidate_fraction=0.6)
problem.setup_cooling_model()
Q, metadata = problem.build_qubo()

# After solving:
problem.set_solution(x_optimal)
stats = problem.get_solution_statistics()
grid = problem.get_solution_grid()
```

---

### 5. `core/problem_modeling/constraint_builder.py`
**Purpose**: High-level constraint construction utilities

**Key Features**:
- Resource constraints (area, budget, water, etc.)
- Cardinality constraints (exact, min, max counts)
- Spatial constraints (connectivity, separation)
- Equity constraints (district fairness, proportional allocation)
- Zoning constraints
- Custom linear constraints
- Solution validation

**Key Classes**:
- `ConstraintSpec`: Constraint specification dataclass
- `ConstraintBuilder`: Main builder class

**Constraint Types Supported**:

1. **Resource Constraints**:
   - `add_resource_constraint()`: Generic resource limit
   - `add_area_constraint()`: Total area limit
   - `add_budget_constraint()`: Total cost limit

2. **Cardinality Constraints**:
   - `add_exact_count_constraint()`: Select exactly k items
   - `add_min_count_constraint()`: At least k items
   - `add_max_count_constraint()`: At most k items

3. **Equity Constraints**:
   - `add_district_equity_constraint()`: Min greens per district
   - `add_proportional_allocation_constraint()`: Proportional to population

4. **Spatial Constraints**:
   - `add_connectivity_constraint()`: Encourage connected clusters
   - `add_separation_constraint()`: Minimum distance between greens

5. **Regulatory Constraints**:
   - `add_zoning_constraint()`: Restrict to allowed zones
   - `add_custom_linear_constraint()`: Arbitrary linear constraints

**Example Usage**:
```python
builder = ConstraintBuilder(n_variables=100)

# Area constraint
builder.add_area_constraint(cell_areas, area_max=50000)

# Budget constraint  
builder.add_budget_constraint(costs, budget_max=1000000)

# Equity: each district gets at least 5 green cells
builder.add_district_equity_constraint(district_ids, min_green_per_district=5)

# Apply to QUBO builder
builder.apply_constraints_to_qubo_builder(qubo_builder)

# Validate solution
validation = builder.validate_solution(x_solution)
```

---

## Integration Example

Here's how all modules work together:

```python
import numpy as np
from core.problem_modeling import (
    UHIGreenLayoutProblem, UHIProblemConfig,
    GaussianCoolingKernel, ConstraintBuilder
)
from core.qubo_formulation import IsingConverter

# 1. Configure problem
config = UHIProblemConfig(
    grid_rows=10, grid_cols=10,
    cell_size=50.0,
    cooling_kernel_type='gaussian',
    cooling_length_scale=100.0,
    cooling_amplitude=-2.0,
    area_budget_fraction=0.3
)

# 2. Setup problem
problem = UHIGreenLayoutProblem(config)
problem.setup_uniform_grid(candidate_fraction=0.6)
problem.setup_cooling_model()

# 3. Build QUBO
Q, metadata = problem.build_qubo()

# 4. Convert to Ising (for quantum solvers)
ising = IsingConverter.qubo_to_ising(Q)

# 5. Solve with your solver (placeholder)
# x_solution = your_solver(Q)  # Classical
# x_solution = quantum_solver(ising)  # Quantum

# 6. Validate and analyze
x_test = np.random.randint(0, 2, problem.n_candidates)
problem.set_solution(x_test)
stats = problem.get_solution_statistics()
validation = problem.validate_solution(x_test)

print(f"Energy: {stats['total_energy']:.2f}")
print(f"Feasible: {validation['is_feasible']}")
print(f"Green cells: {stats['n_green_cells']}")
```

---

## File Structure

```
core/
├── qubo_formulation/
│   ├── __init__.py
│   ├── binary_encoding.py          (existing)
│   ├── constraint_penalty.py       (existing)
│   ├── qubo_matrix_builder.py      (NEW - 18.9 KB)
│   └── ising_converter.py          (NEW - 14.2 KB)
│
├── problem_modeling/
│   ├── __init__.py  
│   ├── cooling_kernels.py          (NEW - 18.7 KB)
│   ├── uhi_green_layout.py         (NEW - 21.2 KB)
│   └── constraint_builder.py       (NEW - 23.6 KB)
│
└── utils/
    ├── __init__.py
    ├── math_helpers.py              (existing)
    ├── matrix_operations.py         (existing)
    └── experiment_logger.py         (existing)
```

---

## Mathematical Foundations

All modules are backed by rigorous mathematical theory:

1. **QUBO Formulation**: Standard binary optimization with quadratic penalty method
2. **Theorem IV.1**: Penalty weight sufficiency for exact constraint embedding
3. **QUBO-Ising Equivalence**: Established transformation via `x = (1+s)/2`
4. **Spatial Kernels**: Based on Gaussian processes and diffusion models
5. **UHI Physics**: Simplified cooling interaction model from urban climatology

---

## Testing

Integration test successfully demonstrated:
- Problem setup with 4×4 grid
- QUBO construction with area constraints
- Ising conversion with verified equivalence (tolerance: 9.54e-07)
- Solution evaluation and validation

---

## Next Steps

To complete the framework, you'll need:

1. **Solvers**:
   - Classical: Simulated Annealing, Tabu Search, Genetic Algorithms
   - Quantum: Simulated Quantum Annealing, D-Wave interface

2. **Validation**:
   - Unit tests for each module
   - Integration tests for workflows
   - Benchmark test cases

3. **Visualization**:
   - Solution grid plotting
   - Convergence curves
   - Temperature heatmaps

4. **Documentation**:
   - API reference
   - Tutorial notebooks
   - Mathematical derivations

---

## Key References

- Section II: QUBO and Ising preliminaries
- Section IV: Quadratic penalty method and constraint embedding
- Section V: UHI-Green-Layout problem definition
- Section VII: Simulated Quantum Annealing formulation

