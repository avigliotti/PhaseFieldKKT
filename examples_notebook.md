# PhaseFieldKKT — Examples

This notebook reproduces the numerical examples from the accompanying paper:
uniaxial dogbone tension tests, compact-specimen pin loading, and periodic
composite RVE analysis under multiaxial macroscopic strain. See the
[README](./README.md) for the theoretical background and the full solver
API (`generic_solver`, `dogbone_solver`, `compactspecimen_solver`,
`solve_periodic_bc`), and the docstring of each solver (e.g.
`?dogbone_solver` in the REPL) for the complete list of keyword arguments.

Install the package directly from GitHub (not yet registered):

```julia
using Pkg
Pkg.add(url="https://github.com/avigliotti/PhaseFieldKKT")
```

```julia
using PhaseFieldKKT
```

`using PhaseFieldKKT` brings `AD4SM`, `Materials`, `Elements`, `Solvers`, and
`PhaseFieldSolvers` into scope, so the material constructors used below
(`Materials.Hooke2D`, `Materials.PhaseField`) and the solver functions
(`PhaseFieldSolvers.dogbone_solver`, etc.) are both available without any
further imports.


## dogbone


```julia
mat = let
         l0,E,ν,σd  = 0.500,430.000,0.170,6.500e-01
         Gc  = 2*l0/E*σd^2
         mat = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
         Materials.PhaseField{typeof(mat),:ATn}(l0, Gc, mat, 1)
       end
data_nhATn1_alt = PhaseFieldSolvers.dogbone_solver(
                        mat       = mat,
                        bwithhist = false,
                        sPostFix  = "nhATn1")
;
```


```julia
mat = let
         l0,E,ν,σd  = 0.500,430.000,0.170,6.500e-01
         Gc  = 2*l0/E*σd^2
         mat = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
         Materials.PhaseField{typeof(mat),:ATn}(l0, Gc, mat, 2)
       end
data_nhATn1_alt = PhaseFieldSolvers.dogbone_solver(
                        mat       = mat,
                        bwithhist = false,
                        sPostFix  = "nhATn2")
;
```


```julia
mat = let
         l0,E,ν,σd  = 0.500,430.000,0.170,6.500e-01
         Gc  = 2*l0/E*σd^2
         mat = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
         Materials.PhaseField{typeof(mat),:ATn}(l0, Gc, mat, 1)
       end
data_nhATn1_alt = PhaseFieldSolvers.dogbone_solver(
                        mat       = mat,
                        bwithhist = true,
                        sPostFix  = "whATn1")
;
```


```julia
mat = let
         l0,E,ν,σd  = 0.500,430.000,0.170,6.500e-01
         Gc  = 2*l0/E*σd^2
         mat = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
         Materials.PhaseField{typeof(mat),:ATn}(l0, Gc, mat, 2)
       end
data_nhATn1_alt = PhaseFieldSolvers.dogbone_solver(
                        mat       = mat,
                        bwithhist = true,
                        sPostFix  = "whATn2")
;
```

## compact test specimen


```julia
mat = let
         l0,E,ν,σd  = 0.500,430.000,0.170,6.500e-01
         Gc  = 2*l0/E*σd^2
         mat = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
         Materials.PhaseField{typeof(mat),:ATn}(l0, Gc, mat, 1)
       end
PhaseFieldSolvers.compactspecimen_solver(
                    mat       = mat,
                    bwithhist = false,
                    sPostFix  = "nhATn1")
;
```


```julia
mat = let
         l0,E,ν,σd  = 0.500,430.000,0.170,6.500e-01
         Gc  = 2*l0/E*σd^2
         mat = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
         Materials.PhaseField{typeof(mat),:ATn}(l0, Gc, mat, 2)
       end
PhaseFieldSolvers.compactspecimen_solver("CTSpecimen_lc5000lci1000", 
                    mat       = mat,
                    bwithhist = false,
                    sPostFix  = "nhATn2")
;
```


```julia
mat = let
         l0,E,ν,σd  = 0.500,430.000,0.170,6.500e-01
         Gc  = 2*l0/E*σd^2
         mat = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
         Materials.PhaseField{typeof(mat),:ATn}(l0, Gc, mat, 1)
       end
PhaseFieldSolvers.compactspecimen_solver("CTSpecimen_lc5000lci1000", 
                    mat       = mat,
                    bwithhist = true,
                    sPostFix  = "whATn1")
;
```


```julia
mat = let
         l0,E,ν,σd  = 0.500,430.000,0.170,6.500e-01
         Gc  = 2*l0/E*σd^2
         mat = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
         Materials.PhaseField{typeof(mat),:ATn}(l0, Gc, mat, 2)
       end
PhaseFieldSolvers.compactspecimen_solver("CTSpecimen_lc5000lci1000", 
                    mat       = mat,
                    bwithhist = true,
                    sPostFix  = "whATn2")
;
```

## periodic boundary conditions


```julia
PhaseFieldSolvers.solve_periodic_bc(
                  ϵM0        = [1, NaN, NaN]*2.500e-03, 
                  θ          = 0.000000, 
                  fiber_mat  = let
                    l0,Ef,νf,ϵdf  = 0.5000,588.00,0.27,6.497e-03
                    Gcf = 2Ef*l0*ϵdf^2
                    mat = Materials.Hooke2D(Ef, νf, small=true, plane_stress=true)
                    Materials.PhaseField{typeof(mat),:ATn}(l0, Gcf, mat, 1)
                  end,
                  matrix_mat = let
                    l0,Em,νm,ϵdm  = 0.5000,430.00,0.17,1.512e-03
                    Gcm = 2Em*l0*ϵdm^2
                    mat = Materials.Hooke2D(Em, νm, small=true, plane_stress=true)
                    Materials.PhaseField{typeof(mat),:ATn}(l0, Gcm, mat, 1)
                  end,
                  bwithhist  = false,
                  sPostFix   = "nhAT1e11NaNNaNt")
;
```


```julia
PhaseFieldSolvers.solve_periodic_bc(
          ϵM0        = [-0.500, 0.866, NaN]*2.500e-03, 
          θ          = 0.000000, 
          fiber_mat  = let
            l0,Ef,νf,ϵdf  = 0.5000,588.00,0.27,6.497e-03
            Gcf = 2Ef*l0*ϵdf^2
            mat = Materials.Hooke2D(Ef, νf, small=true, plane_stress=true)
            Materials.PhaseField{typeof(mat),:ATn}(l0, Gcf, mat, 1)
          end,
          matrix_mat = let
            l0,Em,νm,ϵdm  = 0.5000,430.00,0.17,1.512e-03
            Gcm = 2Em*l0*ϵdm^2
            mat = Materials.Hooke2D(Em, νm, small=true, plane_stress=true)
            Materials.PhaseField{typeof(mat),:ATn}(l0, Gcm, mat, 1)
          end,
          bwithhist  = false,
          sPostFix   = "nhAT1e11e22NaN20")
;
```
