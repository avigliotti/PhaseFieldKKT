# Phase-Field Fracture Solver with Direct Constraint Enforcement

This repository contains the Julia source code accompanying the [research paper](https://rdcu.be/fBIRj):

**"Systematic Analysis of History Field Limitations in Phase-Field Fracture: Direct Constraint Enforcement and Enhanced Energy Decomposition"** by A. Vigliotti and F. Auricchio.

The code implements a finite element framework, built on top of the [AD4SM](https://github.com/avigliotti/AD4SM) automatic-differentiation FE toolkit, for simulating fracture propagation in brittle materials using the phase-field approach, with particular focus on rigorous damage irreversibility constraint enforcement.

## Theoretical Background

Phase-field fracture models regularize sharp crack surfaces using a continuous damage field $d \in [0,1]$, where $d=0$ represents intact material and $d=1$ represents fully damaged material. The evolution is governed by the minimization of a total energy functional:

$$E(\mathbf{u}, d) = \int_{\Omega} \left[ \phi^- + (1-d)^2\phi^+ + \frac{G_\textrm{c}}{2l_0} \left( d^p + l_0^2 d_{,i}d_{,i} \right) \right] d\Omega$$

where:
*   $\mathbf{u}$: Displacement field
*   $d$: Scalar damage phase-field
*   $\phi^+$, $\phi^-$: Tensile and compressive parts of the elastic strain energy density
*   $G_c$: Material fracture toughness
*   $l_0$: Length scale parameter controlling the regularization width
*   $p$: Model exponent, defining AT1 ($p=1$) or AT2 ($p=2$) formulations

## Key Research Contributions

This work addresses critical limitations in damage irreversibility constraint enforcement ($\dot{d} \geq 0$) through:

### 1. Analysis of History Field Method Limitations
Traditional **history-field** methods, which store the maximum tensile energy $\phi^+$ over time, are shown to be mathematically equivalent to truncating negative damage increments. This approach:
*   Works well for AT2 models with smooth damage evolution
*   Introduces systematic errors in AT1 models (15-25% strength overestimation)
*   Violates the smoothness requirements of the underlying variational formulation
*   Creates artificial slope discontinuities that compromise gradient regularization

### 2. Direct Constraint Enforcement via KKT Conditions
A parameter-free active set algorithm directly enforces the Karush-Kuhn-Tucker (KKT) conditions for the inequality-constrained optimization problem. This approach:
*   Eliminates systematic errors in AT1 models
*   Maintains full variational consistency
*   Preserves compatibility with AT2 formulations
*   Provides more reliable predictions under complex stress states

### 3. Enhanced λ-μ Strain Energy Decomposition
An improved strain energy decomposition that:
*   Eliminates the infinite strength singularity under hydrostatic compression
*   Satisfies four of five theoretical requirements established by recent literature
*   Maintains computational efficiency without additional parameters
*   Enables damage evolution under all loading conditions

## Installation

This package is not (yet) registered; install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/avigliotti/PhaseFieldKKT")
```

```julia
using PhaseFieldKKT
```

`using PhaseFieldKKT` re-exports `AD4SM`, `Materials`, `Elements`, `Solvers`,
and `PhaseFieldSolvers`, so material and element constructors
(`Materials.Hooke2D`, `Materials.PhaseField`, `Elements.TriaP`, …) and the
solvers described below are all available without further `using`/`import`
statements.

## Solver Capabilities

Four solver entry points are provided in `PhaseFieldSolvers`, each running
the same staggered displacement/damage (`u`/`d`) Newton–KKT scheme, differing
in how the mesh, boundary conditions, and (where relevant) periodicity or
loading constraints are set up:

1.  **`generic_solver(nodes, elems, bc_u; kwargs...)`** — the underlying
    staggered `u`/`d` solver, taking an already-built mesh and a
    displacement boundary-condition array directly. Use this for custom
    geometries/boundary conditions not covered by the specialized solvers
    below.

2.  **`dogbone_solver(sModelName; kwargs...)`** — uniaxial tensile test on a
    dogbone specimen, loaded via prescribed displacement on `"top"`/`"btm"`
    node sets. This is the solver used to generate the strength-comparison
    results (history-field vs. direct KKT, AT1 vs. AT2) in the paper.

3.  **`compactspecimen_solver(sModelName; kwargs...)`** — fracture test on a
    specimen with circular loaded boundaries (e.g. compact-tension-style
    pin loading), enforced via nonlinear constraint equations rather than
    direct displacement boundary conditions.

4.  **`solve_periodic_bc(sModelName; kwargs...)`** — fracture propagation in
    a periodic composite (fiber/matrix) representative volume element under
    a prescribed macroscopic strain history, with periodicity enforced
    through explicit constraint matrices and reduced-system Pardiso solves.

See each function's docstring (`?dogbone_solver`, etc., from the Julia REPL)
for the full list of keyword arguments, defaults, and return values.

### Key Features
*   **Constraint Enforcement**: compare history-field vs. direct KKT methods (the `bwithhist` keyword, common to all four solvers)
*   **Energy Decompositions**: traditional hydrostatic-deviatoric vs. enhanced λ-μ decomposition
*   **Material Models**: AT1 and AT2 phase-field formulations (`Materials.PhaseField{...,:ATn}`, selected via the model exponent `n`), with detailed error analysis in the paper
*   **Element Support**: 2D and 3D phase-field elements (`TriaP`, `QuadP`, `Tet04P`, `Hex08P`, `Wdg06P`, plus assumed-strain `ASTria`/`ASQuad` variants)
*   **Automatic Differentiation**: element residuals and tangents (both mechanical and damage) computed via forward-mode dual numbers, not hand-derived
*   **Input/Output**: Abaqus `.inp` mesh files for input; VTK/ParaView collections and JLD2 binary archives for output

## Numerical Examples

See [`./examples_notebook.md`](./examples_notebook.md) or [`./examples_notebook.ipynb`](./examples_notebook.ipynb) for
runnable examples covering:

*   **Uniaxial tensile tests** (`dogbone_solver`) — comparing AT1 vs. AT2
    strength predictions, with and without the history-field approximation.
*   **Compact tension specimens** (`compactspecimen_solver`) — validating
    fracture toughness measurements under pin loading.
*   **Composite RVE analysis** (`solve_periodic_bc`) — multiaxial loading
    effects on fracture in a periodic fiber/matrix microstructure, under
    different macroscopic strain states.

These examples systematically demonstrate when history-field methods are
adequate (AT2 models) and when direct constraint enforcement becomes
necessary (AT1 models, complex/multiaxial loading), reproducing the
comparisons presented in the paper.

## Citation

If you use this code in your research, please cite:
```bibtex
@ARTICLE{Vigliotti2026,
	author = {Vigliotti, Andrea and Auricchio, Ferdinando},
	title = {Systematic analysis of damage irreversibility enforcement in phase-field fracture models},
	year = {2026},
	journal = {International Journal of Fracture},
	volume = {250},
	number = {4},
	doi = {10.1007/s10704-026-00940-z},
	type = {Article}
}
```
## License

This software is released under the MIT License. See LICENSE file for details.

Academic and commercial use is permitted. If you use this code in your research, please cite the accompanying paper as shown above.
