module PhaseFieldSolvers

using LinearAlgebra, SparseArrays, Printf, Random
# using AbaqusReader, Logging, FileIO, Dates
using WriteVTK, FileIO, Dates
using Pardiso

using AD4SM
using  .Solvers, .Materials, .Elements

export generic_solver, compactspecimen_solver, dogbone_solver, periodicbc_solver

using ..HelperFuncs

"""
    generic_solver(nodes, elems, bc_u; kwargs...) -> Dict

Solve a displacement-controlled phase-field fracture problem on an arbitrary
mesh, using a staggered scheme: at each load step, the displacement field
`u` is solved to mechanical equilibrium for the current damage field `d`
(via [`update_u!`](@ref)), then the damage field is updated for the current
displacement field (via [`update_d!`](@ref)), and the two are iterated until
both converge or `maxiter` is reached. This is the most general of the four
solvers — the specimen-specific solvers (`dogbone_solver`,
`compactspecimen_solver`) build their own `nodes`/`elems`/`bc_u` from a named
mesh and then largely repeat this same loop with problem-specific boundary
conditions and post-processing.

# Arguments
- `nodes`: vector of nodal reference coordinates, one entry per node.
- `elems`: vector of phase-field elements (`CPElem`, e.g. as returned by
  `Elements.TriaP`/`Elements.QuadP`/`Elements.Tet04P`/`Elements.Hex08P`).
- `bc_u::Array`: displacement boundary-condition array, same shape as the
  nodal displacement field `u` (`nDoFs × nNodes`). Entries equal to `NaN`
  mark a free DOF; any other value is a target displacement that is ramped
  linearly from `0` to that value over the load steps (`u = bc_u .* LF`,
  `LF` running from `0` to `1`).

# Keyword Arguments
- `sModelName = ""`: model name, used only to build output file names —
  does **not** load a mesh (the mesh must already be built into `nodes`,
  `elems` before calling this function).
- `sMeshPath = "./meshes"`: unused by this method directly, kept for
  interface parity with the specimen-specific solvers.
- `svtkPath = "./vtk_files"`: output directory for `.pvd`/VTK state files
  (created automatically if it does not exist).
- `sJLD2Path = "./jld2_files"`: output directory for the final `.jld2`
  results file (created automatically if it does not exist).
- `dTolΔd = 5e-3`: staggered-scheme convergence tolerance on the maximum
  change in the damage field `d` between successive `(u,d)` sub-iterations
  within a load step.
- `maxiter = 10`: maximum number of staggered `u`/`d` sub-iterations per
  load step before the step is abandoned as failed.
- `isave = 5`: interval (in load steps) at which the full `(u,d,ru)` state
  is appended to the in-memory `allus` history (returned in the result
  dictionary; independent of VTK/JLD2 output).
- `nwritevtu = isave`: interval (in load steps) at which a VTK state is
  written to the ParaView collection. Pass `NaN` to disable VTK output
  entirely (checked via `isnan(nwritevtu)` before the collection is even
  opened).
- `nSteps = 50`: number of load steps; the load factor `LF` is sampled
  uniformly on `[0,1]` with this many points.
- `bwithhist = false`: selects the damage-update strategy. `false` uses the
  direct KKT/active-set enforcement of damage irreversibility (no history
  variable is tracked); `true` uses the classical history-field method
  (tracks, per Gauss point, the maximum tensile energy density seen so far).
- `becho = false`: print per-iteration Newton/KKT diagnostics
  (residual/update extrema) from `update_u!`/`update_d!`.
- `bdef = false`: write the *deformed* configuration (`nodes + u`) rather
  than the reference configuration in VTK output.
- `sPostFix = "nhAT2"`: string appended to `sModelName` to form the output
  file base name (VTK collection and `.jld2` file).

# Returns
A `Dict` with (at least) the following keys:
- `"results"`: `Dict("u" => ..., "ru" => ...)`, per-step converged
  displacement and reaction-force fields restricted to the constrained
  DOFs (as sparse arrays, one per load step).
- `"Vd"`: vector of the load-step-averaged damaged-volume fraction
  `∫d dΩ / ∫dΩ`, one entry per step.
- `"allus"`: list of `(u, d, ru)` snapshots saved every `isave` steps.
- `"niters"`, `"LF"`, `"dTolΔd"`, `"maxiter"`, `"nSteps"`, `"bc_u"`,
  `"bwithhist"`, `"sPostFix"`, `"isave"`, `"sModelName"`: run parameters
  and diagnostics, saved verbatim to the `.jld2` file for reproducibility.

If a load step fails to converge (`update_u!` returns `false`) or an
exception is raised mid-run, the loop stops early, the last VTK/JLD2 state
is still written, and the returned dictionary is truncated to the steps
actually completed — the function does not otherwise re-throw.

# Example
```julia
mat  = Materials.PhaseField{typeof(hooke), :ATn}(l0, Gc, hooke, 2)
u_bc = fill(NaN, 2, length(nodes))
u_bc[2, top_nodes]    .=  0.1
u_bc[2, bottom_nodes] .= -0.1
results = generic_solver(nodes, elems, u_bc; nSteps=100, bwithhist=false)
```
"""
function generic_solver(nodes, elems, bc_u::Array;
                        sModelName = "",
                        sMeshPath  = "./meshes",
                        svtkPath   = "./vtk_files",
                        sJLD2Path  = "./jld2_files",
                        dTolΔd     = 5e-3,
                        maxiter    = 10,
                        isave      = 5,
                        nwritevtu  = isave,
                        nSteps     = 50,
                        bwithhist  = false,
                        becho      = false,
                        bdef       = false,
                        sPostFix   = "nhAT2")

  @show sMeshPath
  println("Model name: ", sModelName*".inp")

  # Ensure output directories exist
  mkpath(svtkPath)
  mkpath(sJLD2Path)

  sFileName     = @sprintf("%s%s", sModelName, sPostFix)
  spvdFileName  = joinpath(svtkPath,sFileName)

  println("\nstarting ", sFileName, " on ", now())

  nNodes  = length(nodes)
  u       = zero(bc_u)
  ru      = zero(bc_u)
  d       = zeros(1, nNodes)
  rd      = zeros(nNodes)
  uprev   = copy(u)
  dprev   = copy(d)

  ϕmax    = if bwithhist
    [[(0., 0.) for ii in 1:length(elem.wgt)] for elem in elems]
  else
    nothing
  end
  @show typeof(ϕmax)

  bfree   = isnan.(bc_u)
  bcnst   = .!bfree
  ifree   = findall(bfree[:]) 

  LF      = range(0,1,length=nSteps)
  allus   = []
  results = Dict("u"  => [spzeros(size(bc_u)) for ii=1:nSteps],
                 "ru" => [spzeros(size(bc_u)) for ii=1:nSteps])

  Vol     = sum(elem->elem.V, elems)
  Vd      = fill(NaN, nSteps) 
  niters  = zeros(Int, nSteps)

  bfailed = false
  println("\n\tstarting solver")
  tstart  = Base.time_ns()
  Δd      = nothing

  if !isnan(nwritevtu)
    paraview_collection(spvdFileName) do pvd; end
  end

  iStep = 1
  try
    for (ii,LF)  = enumerate(LF)
      printstyled("\nstarting step $ii\n", color=:blue)
      t0         = Base.time_ns()

      copyto!(uprev, u)
      u[bcnst]   = bc_u[bcnst]*LF
      for iter=1:maxiter
        niters[ii] = iter
        if !update_u!(elems, u, d, ifree, ru, becho=becho)
          @printf("\tFailed at LF: %.4f, step %-3i\n", LF, ii)
          bfailed = true
          break
        end
        copyto!(dprev, d)
        rd .= update_d!(elems, u, d, ϕmax, becho=becho)

        Δd    = extrema(d-dprev)
        @printf("iter %2i done, Δd: %0.4f … %0.4f → d: %0.4f … %0.4f\n", 
                iter, Δd..., extrema(d)...)
        maximum(Δd) < dTolΔd && break    
      end
      stptime    = (Base.time_ns()-t0)/1e9
      tottime    = (Base.time_ns()-tstart)/1e9/60
      bfailed && break

      @printf("LF: %.4f, step %-3i done in %2i iters, %.3f sec; el. time %.2f mins, ETA %.2f mins \n",
              LF, ii, niters[ii], stptime, tottime, stptime*(nSteps-ii)/60)

      results["u"][ii][bcnst]  = u[bcnst]
      results["ru"][ii][bcnst] = ru[bcnst]
      Vd[ii]                   = getVd(elems, d)/Vol

      if ii % isave == 1
        push!(allus, (copy(u), copy(d), copy(ru)))
        println("--- storing step: ", ii) 
      end

      if ii % nwritevtu == 0      
        paraview_collection(spvdFileName; append=true) do pvd
          item = writeVTKstate(spvdFileName, nodes, elems, u, 
                               elemsprop=get_ϵT(elems, u),
                               nodesprop=Dict("d"=>d, "ddot"=>d-dprev,
                                              "u"=>u, "udot"=>u-uprev, 
                                              "ru"=>ru, "rd"=>rd), 
                               pvd=pvd, bdef=bdef, bcenter=false,ii=ii)
          println(" state written to: ", item)
        end
      end  

      iStep = ii
      flush(stdout)
    end
  catch e
    u[:]    .= uprev[:]
    d[:]    .= dprev[:]

    results = Dict("u"  => results["u"][1:iStep],
                   "ru" => results["ru"][1:iStep])
    LF      = LF[1:iStep]
    niters  = niters[1:iStep]
    ##
    if !isnan(nwritevtu)
      paraview_collection(spvdFileName; append=true) do pvd
        item = writeVTKstate(spvdFileName, nodes, elems, u, 
                             elemsprop=get_ϵT(elems, u),
                             nodesprop=Dict("d"=>d, "ddot"=>d-dprev,
                                            "u"=>u, "udot"=>u-uprev, 
                                            "ru"=>ru, "rd"=>rd), 
                             pvd=pvd, bdef=bdef, bcenter=false,ii=iStep)
        println(" state written to: ", item)
      end
    end
    ##

    error_msg = sprint(showerror, e)
    st        = sprint((io,v) -> show(io, "text/plain", v), 
                       stacktrace(catch_backtrace()))
    @warn "Trouble doing things:\n$(error_msg)\n$(st)" 
    println(" quitting." ); flush(stdout)
  end

  saved_vars = Dict("sModelName"=>sModelName, "dTolΔd"=>dTolΔd,
                    "maxiter"=>maxiter, "isave"=>isave, "nSteps"=>nSteps,
                    "allus"=>allus,     "bc_u"=>bc_u,   "niters"=>niters, 
                    "LF"=>LF,     "sPostFix"=>sPostFix, "bwithhist"=>bwithhist,
                    "results"=>results, "Vd"=>Vd,
                    "isave"=>isave)

  let sFileName = joinpath(sJLD2Path,sFileName*".jld2")
    println("\n saving binary data to: ", sFileName, " ..."); flush(stdout)
    FileIO.save(sFileName, saved_vars)
  end

  let item = writeVTKstate(spvdFileName, nodes, elems, u, 
                           elemsprop=get_ϵT(elems, u),
                           nodesprop=Dict("d"=>d, "u"=>u), 
                           bdef=bdef, bcenter=false)
    println(" last state written to: ", item)
  end
  @printf("\n\nall done in %s\n", secs2hms((Base.time_ns()-tstart)/1e9))
  @show now()

  return saved_vars
end

"""
    dogbone_solver(sModelName="dogbone_tria0600h0030b0050lci"; kwargs...) -> Dict

Solve a uniaxial tensile ("dogbone") specimen fracture test: loads an Abaqus
mesh named `sModelName.inp` from `sMeshPath`, applies symmetric, opposite
vertical displacement to node sets `"top"` and `"btm"` (and, if present,
constrains the horizontal displacement of a `"cntr"` node set to suppress
rigid-body motion), and runs the same staggered `u`/`d` solve as
[`generic_solver`](@ref), tracking the load-displacement curve as it goes.
This is the specimen used throughout the accompanying paper to compare
strength predictions between the history-field and direct-KKT damage
irreversibility enforcement methods.

# Arguments
- `sModelName = "dogbone_tria0600h0030b0050lci"`: base name of the Abaqus
  `.inp` mesh file (without extension), searched for in `sMeshPath`. The
  mesh must define node sets `"top"` and `"btm"`, and may optionally define
  a `"cntr"` node set (fixed against horizontal drift).

# Keyword Arguments
- `sMeshPath = joinpath(pkgdir(@__MODULE__), "meshes")`: directory
  containing the mesh file; defaults to the meshes bundled with the package.
- `svtkPath = "./vtk_files"` / `sJLD2Path = "./jld2_files"`: output
  directories for VTK and `.jld2` results (created automatically).
- `mat`: the `PhaseField` material model (default: an AT2 `Hooke2D`
  plane-strain material with `l0=0.5`, `E=430`, `ν=0.17`,
  `ϵd=1.512e-3`, `n=2` — see source for exact construction). Pass your own
  `Materials.PhaseField{...}` instance to change the constitutive law,
  length scale, fracture toughness, or AT1 (`n=1`) vs. AT2 (`n=2`) model.
- `Δumax = 1/10`: total (symmetric) relative displacement applied between
  the `"top"` and `"btm"` node sets at load factor `1` (i.e. `"top"` moves
  by `+Δumax/2`, `"btm"` by `-Δumax/2`).
- `dTolΔd = 1e-2`: staggered-scheme convergence tolerance on the maximum
  change in `d` between sub-iterations.
- `dTold = 1e-6`: convergence tolerance passed through to the damage
  sub-solver (`update_d!`) itself, distinct from the staggered-scheme
  tolerance `dTolΔd` above.
- `maxiter = 20`: maximum staggered `u`/`d` sub-iterations per load step
  (also reused as the damage sub-solver's own `maxiter`, unless overridden).
- `isave = 5`, `nwritevtu = isave`: see [`generic_solver`](@ref).
- `nSteps = 250`: number of load steps.
- `bisAS = false`: use assumed-strain (`ASQuad`/`ASTria`) elements instead
  of the standard displacement-based phase-field elements (`QuadP`/`TriaP`).
- `bwithhist = false`, `becho = false`, `bdef = false`: see
  [`generic_solver`](@ref).
- `sPostFix = string(bwithhist ? "wh" : "nh", "ATn", mat.n)`: output file
  suffix; by default encodes whether history-field (`wh`) or direct-KKT
  (`nh`) enforcement was used, and the AT-model exponent `mat.n`.

# Returns
A `Dict` including, in addition to the keys documented in
[`generic_solver`](@ref):
- `"results"`: `Dict("u" => ..., "ru" => ...)` where, unlike
  `generic_solver`, `"u"` is the scalar top-minus-bottom mean vertical
  displacement at each step (the effective specimen elongation) and `"ru"`
  is the summed vertical reaction force on the `"top"` node set — i.e. this
  is already reduced to a load–displacement curve, ready for plotting.
- `"Δumax"`: the applied maximum displacement, echoed back for convenience.

# Example
```julia
mat = let
    l0, E, ν, σd = 0.5, 430.0, 0.17, 0.65
    Gc = 2*l0/E*σd^2
    hooke = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
    Materials.PhaseField{typeof(hooke),:ATn}(l0, Gc, hooke, 2)  # AT2
end
data = dogbone_solver(mat=mat, bwithhist=false, sPostFix="nhATn2")
```
"""
function dogbone_solver(sModelName="dogbone_tria0600h0030b0050lci";
                       sMeshPath = joinpath(pkgdir(@__MODULE__), "meshes"),
                       svtkPath  = "./vtk_files",
                       sJLD2Path = "./jld2_files",
                       mat       = let
                         l0,Em,νm,ϵdm  = 0.5000,430.00,0.17,1.512e-03
                         Gcm = 2Em*l0*ϵdm^2
                         PhaseField(l0, Gcm, Hooke2D(Em, νm, small=true, plane_stress=false), 2)
                       end,
                       Δumax      = 1/10,
                       dTolΔd     = 1e-2,
                       dTold      = 1e-6,
                       maxiter    = 20,
                       isave      = 5,
                       nwritevtu  = isave,
                       nSteps     = 250,
                       bisAS      = false,
                       bwithhist  = false,
                       becho      = false,
                       bdef       = false,
                       sPostFix   = string(bwithhist ? "wh" : "nh", "ATn", mat.n))

  @show sMeshPath
  println("Model name: ", sModelName*".inp")

  # Ensure output directories exist
  mkpath(svtkPath)
  mkpath(sJLD2Path)

  mean(x) = sum(x)/length(x)

  sFileName     = @sprintf("%s%s", sModelName, sPostFix)
  spvdFileName  = joinpath(svtkPath,sFileName)

  println("\nstarting ", sFileName, " on ", now(), "\n")

  @show Δumax
  dump(mat)

  nodes, elems,
  node_sets, elem_sets = make_the_2Dmodel(joinpath(sMeshPath, sModelName), 
                                          mat, bisAS=bisAS)

  nNodes = length(nodes)
  bc_u   = fill(NaN, 2, nNodes)
  bc_u[1, node_sets["top"]] .= 0
  bc_u[2, node_sets["top"]] .= Δumax/2
  bc_u[1, node_sets["btm"]] .= 0
  bc_u[2, node_sets["btm"]] .= -Δumax/2

  if haskey(node_sets, "cntr") # && bisAS
    bc_u[1, node_sets["cntr"]] .= 0
    @show haskey(node_sets, "cntr")
  end

  nNodes  = length(nodes)
  u       = zero(bc_u)
  ru      = zero(bc_u)
  d       = zeros(1, nNodes)
  rd      = zeros(nNodes)
  uprev   = copy(u)
  dprev   = copy(d)

  ϕmax    = if bwithhist
    [[(0., 0.) for ii in 1:length(elem.wgt)] for elem in elems]
  else
    nothing
  end
  @show typeof(ϕmax)

  bfree   = isnan.(bc_u)
  bcnst   = .!bfree
  ifree   = findall(bfree[:]) 

  LF      = range(0,1,length=nSteps)
  allus   = []
  results = Dict("u"  => fill(NaN, nSteps),
                 "ru" => fill(NaN, nSteps))

  Vol     = sum(elem->elem.V, elems)
  Vd      = fill(NaN, nSteps) 
  niters  = zeros(Int, nSteps)

  bfailed = false
  println("\n\tstarting solver")
  tstart  = Base.time_ns()
  Δd      = nothing

  if !isnan(nwritevtu)
    paraview_collection(spvdFileName) do pvd; end
  end

  iStep = 1
  try
    while iStep ≤ nSteps
      printstyled("\nstarting step $iStep/$nSteps, LF: ", 
                  round(LF[iStep],sigdigits=3), "\n", color=:blue)
      t0         = Base.time_ns()

      copyto!(uprev, u)
      u[bcnst]   = bc_u[bcnst]*LF[iStep]
      for iter=1:maxiter
        niters[iStep] = iter
        if !update_u!(elems, u, d, ifree, ru, becho=becho)
          @printf("\tFailed at LF: %.4f, step %-3i\n", LF[iStep], iStep)
          bfailed = true
          break
        end
        copyto!(dprev, d)
        rd[:]  = update_d!(elems, u, d, ϕmax, 
                           dTol=dTold, maxiter=maxiter, becho=becho)
        Vditer = getVd(elems, d)/Vol
        Δd     = d-dprev
        @printf("iter %2i done, Δd: %0.4f … %0.4f → d: %0.4f … %0.4f\n", 
                iter, extrema(Δd)..., extrema(d)...)
        maximum(Δd) < dTolΔd && break    
      end
      stptime    = (Base.time_ns()-t0)/1e9
      tottime    = (Base.time_ns()-tstart)/1e9
      bfailed && break

      @printf("\tΔd: %0.4f … %0.4f → d: %0.4f … %0.4f\n", extrema(Δd)..., extrema(d)...)
      println("Step $iStep/$nSteps completed in ", round(stptime / 60, digits = 2), " mins., ",
              "Elapsed time ", secs2hms(tottime), ", ",
              "ETA ", secs2hms((nSteps - iStep) * stptime), "\n")

      results["u"][iStep]  = mean(u[2,node_sets["top"]])-mean(u[2,node_sets["btm"]])
      results["ru"][iStep] = sum(ru[2,node_sets["top"]])
      Vd[iStep]            = getVd(elems, d)/Vol

      if iStep % isave == 1
        push!(allus, (copy(u), copy(d), copy(ru)))
        println("--- storing step: ", iStep) 
      end

      if iStep % nwritevtu == 0      
        paraview_collection(spvdFileName; append=true) do pvd
          item = writeVTKstate(spvdFileName, nodes, elems, u, 
                               elemsprop=get_ϵT(elems, u),
                               nodesprop=Dict("d"=>d, "ddot"=>d-dprev,
                                              "u"=>u, "udot"=>u-uprev, 
                                              "ru"=>ru, "rd"=>rd), 
                               pvd=pvd, bdef=bdef, bcenter=false,ii=iStep)
          println(" state written to: ", item)
        end
      end  

      iStep += 1
      flush(stdout)
    end
  catch e
    u[:]    .= uprev[:]
    d[:]    .= dprev[:]

    results = Dict("u"  => results["u"][1:iStep],
                   "ru" => results["ru"][1:iStep])
    LF      = LF[1:iStep]
    niters  = niters[1:iStep]

    error_msg = sprint(showerror, e)
    flush(stdout)
    st        = sprint((io, v) -> show(io, "text/plain", v),
                       stacktrace(catch_backtrace()))
    @warn "Simulation error at step $iStep:\n$(error_msg)\n$(st)"
    println("Quitting.")
    flush(stdout)
  end

  saved_vars = Dict("sModelName"=>sModelName, "dTolΔd"=>dTolΔd,
                    "maxiter"=>maxiter, "isave"=>isave, "nSteps"=>nSteps,
                    "allus"=>allus,     "bc_u"=>bc_u,   "niters"=>niters, 
                    "LF"=>LF,     "sPostFix"=>sPostFix, "bwithhist"=>bwithhist,
                    "results"=>results, "Vd"=>Vd, "Δumax"=>Δumax,
                    "isave"=>isave)

  let sFileName = joinpath(sJLD2Path,sFileName*".jld2")
    println("\n saving binary data to: ", sFileName, " ..."); flush(stdout)
    FileIO.save(sFileName, saved_vars)
  end

  let item = writeVTKstate(spvdFileName, nodes, elems, u, 
                           elemsprop=get_ϵT(elems, u),
                           nodesprop=Dict("d"=>d, "u"=>u), 
                           bdef=bdef, bcenter=false)
    println(" last state written to: ", item)
  end

  @printf("\n\nall done in %s\n", secs2hms((Base.time_ns()-tstart)/1e9))
  @show now()

  return saved_vars
end

"""
    compactspecimen_solver(sModelName="CTSpecimentria_b4000stdlc0250lci0050"; kwargs...) -> Dict

Solve a fracture problem on a compact-tension-style specimen loaded through pin holes, 
enforcing the loading via nonlinear constraint equations rather than direct
displacement boundary conditions. The centers of the node sets
`"topcirc"`/`"btmcirc"` are held to move apart by `Δy` (ramped over the load
steps) while every node on those circles is constrained to remain at its
original distance from its respective moving center — modeling a rigid pin
loading a circular hole without prescribing individual nodal displacements.
Displacement equilibrium is solved via the constrained Newton scheme
[`update_u!(elems, eqns, u, d, ru; kwargs...)`](@ref), and damage is updated
exactly as in [`generic_solver`](@ref).

# Arguments
- `sModelName = "CTSpecimentria_b4000stdlc0250lci0050"`: base name of the
  Abaqus `.inp` mesh file, searched for in `sMeshPath`. The mesh must define
  node sets `"topcirc"` and `"btmcirc"` delineating the loaded circular
  boundaries.

# Keyword Arguments
- `sMeshPath = joinpath(pkgdir(@__MODULE__), "meshes")`,
  `svtkPath = "./vtk_files"`, `sJLD2Path = "./jld2_files"`: see
  [`dogbone_solver`](@ref).
- `mat`: `PhaseField` material model (default: AT2 `Hooke2D` plane-strain,
  same defaults as `dogbone_solver`).
- `bwithhist = false`: history-field vs. direct-KKT damage update, as
  elsewhere.
- `dTold = 2.5e-5`: convergence tolerance for the damage sub-solver
  (`update_d!`) itself.
- `dTolΔd = 5e-3`: staggered-scheme convergence tolerance on `d`.
- `maxiter = 10`: maximum staggered `u`/`d` sub-iterations per load step.
- `maxiterd = maxiter`: maximum iterations for the damage sub-solver,
  independent of the staggered-scheme `maxiter` above.
- `isave = 5`, `nwritevtu = isave`: see [`generic_solver`](@ref).
- `nSteps = 50`: number of load steps.
- `Δymax = 80/150`: total relative separation applied between the circle
  centers at load factor `1`.
- `bdef = false`, `becho = false`: see [`generic_solver`](@ref).
- `sPostFix = "nhAT2"`: output file suffix.

# Returns
A `Dict` including:
- `"Rf"`: `2 × nSteps` array, the reaction force (summed over `"topcirc"`)
  at each load step — the load-cell reading for a load–displacement curve.
- `"Δv"`: the applied separation `Δy` at each load step.
- `"Vd"`: damaged-volume fraction per step, as in `generic_solver`.
- `"allus"`: `(u, d)` snapshots saved every `isave` steps.
- `"mat"`, `"dTolΔd"`, `"maxiter"`, `"nSteps"`, `"bwithhist"`, `"sPostFix"`,
  `"ϕmax"`: run parameters and (if `bwithhist=true`) the final history-field
  state, saved for reproducibility.

# Example
```julia
mat = let
    l0, E, ν, σd = 0.5, 430.0, 0.17, 0.65
    Gc = 2*l0/E*σd^2
    hooke = Materials.Hooke2D(E, ν, small=true, plane_stress=true)
    Materials.PhaseField{typeof(hooke),:ATn}(l0, Gc, hooke, 1)  # AT1
end
compactspecimen_solver("CTSpecimen_lc5000lci1000", mat=mat, bwithhist=false)
```
"""
function compactspecimen_solver(sModelName="CTSpecimentria_b4000stdlc0250lci0050";
                                sMeshPath = joinpath(pkgdir(@__MODULE__), "meshes"),
                                svtkPath  = "./vtk_files",
                                sJLD2Path = "./jld2_files",
                                mat       = let
                                  l0,Em,νm,ϵdm  = 0.5000,430.00,0.17,1.512e-03
                                  Gcm = 2Em*l0*ϵdm^2
                                  PhaseField(l0, Gcm, Hooke2D(Em, νm, small=true, plane_stress=false), 2)
                                end,
                                bwithhist = false,
                                dTold     = 2.5e-5,
                                dTolΔd    = 5e-3,
                                maxiter   = 10,
                                maxiterd  = maxiter,
                                isave     = 5,
                                nwritevtu = isave,
                                nSteps    = 50,
                                Δymax     = 80/150,
                                bdef      = false,
                                becho     = false,
                                sPostFix  = "nhAT2")

  @show sMeshPath
  println("Model name: ", sModelName*".inp")

  # Ensure output directories exist
  mkpath(svtkPath)
  mkpath(sJLD2Path)

  normsqrd(x) = x⋅x

  @show now()
  println("\n\nstarting ", sModelName)

  println("\n\n material: ")
  dump(mat)
  println()

  sFileName     = @sprintf("%s%s", sModelName, sPostFix)
  spvdFileName  = joinpath(svtkPath,sFileName)

  nodes, elems,
  node_sets, elem_sets = make_the_2Dmodel(joinpath(sMeshPath, sModelName), mat)

  @show nNodes = length(nodes)
  u      = zeros(2, nNodes)
  ru     = zeros(2, nNodes)
  d      = zeros(1, nNodes)  
  uprev  = copy(u)
  dprev  = copy(d)
  ϕmax        = if bwithhist
    [[(0., 0.) for ii in 1:length(elem.wgt)] for elem in elems]
  else
    nothing
  end
  @show typeof(ϕmax)

  itop   = node_sets["topcirc"]
  ibtm   = node_sets["btmcirc"]

  cgtop  = get_cg(nodes[itop])
  cgbtm  = get_cg(nodes[ibtm])
  dtop0  = [normsqrd(node-cgtop) for node in nodes[itop]]
  dbtm0  = [normsqrd(node-cgbtm) for node in nodes[ibtm]]

  Δy     =  0
  eqns   = vcat([ConstEq(u->normsqrd(nodes[inode]+u-(cgtop+[0, Δy/2]))-dtop0[ii],
                         2inode-1:2inode) for (ii,inode) in enumerate(itop)],
                [ConstEq(u->normsqrd(nodes[inode]+u-(cgbtm+[0,-Δy/2]))-dbtm0[ii],
                         2inode-1:2inode) for (ii,inode) in enumerate(ibtm)])

  LF    = range(0,1,length=nSteps)
  Rf    = zeros(2,nSteps)
  Δv    = zeros(nSteps)
  Vd    = fill(NaN, nSteps) 
  Vol   = sum(elem->elem.V, elems)

  allus = []

  Vdprev  = 0.0
  bfailed = false
  println("\tstarting")
  tstart  = Base.time_ns()

  try
    if !isnan(nwritevtu)
      paraview_collection(spvdFileName) do pvd; end
    end
    for (ii,LF) = enumerate(LF)
      copyto!(uprev, u)

      Δv[ii] = Δy =  Δymax*LF
      printstyled("\nstarting step $ii/$nSteps, LF: ", 
                  round(LF,sigdigits=3), "\n", color=:blue)

      t0         = Base.time_ns()
      for iter=0:maxiter
        if !update_u!(elems, eqns, u, d, ru, becho=becho)
          bfailed = true
          break
        end
        copyto!(dprev, d)
        update_d!(elems, u, d, ϕmax, 
                  dTol=dTold, maxiter=maxiterd, becho=becho)

        @printf("iter %2i done, Δd: %0.4f … %0.4f → d: %0.4f … %0.4f\n", 
                iter, extrema(d-dprev)..., extrema(d)...)
        maximum(d-dprev) < dTolΔd && break    
      end
      stptime    = (Base.time_ns()-t0)/1e9
      tottime     = (Base.time_ns()-tstart)/1e9/60
      bfailed && break

      Rf[:,ii]    = sum(ru[:, itop], dims=2)
      Vd[ii]      = getVd(elems, d)/Vol

      if ii % isave == 1
        push!(allus, (copy(u), copy(d)))
        println("--- storing step: ", ii) 
      end

      if ii % nwritevtu == 0
        paraview_collection(spvdFileName; append=true) do pvd
          item = writeVTKstate(spvdFileName, nodes, elems, u, 
                               elemsprop=get_ϵT(elems, u),
                               nodesprop=Dict("d"=>d, "u"=>u, "Fi"=>ru), 
                               pvd=pvd, bdef=bdef, bcenter=false,ii=ii)
          println(" state written to: ", item)
        end
      end  

      @printf("step %-3i, LF:%.4f, done in %.3f sec; ΔVd:%.2e; el. time: %.2f mins, ETA %.2f mins \n",
              ii, LF, stptime, Vd[ii]-Vdprev, tottime, stptime*(nSteps-ii)/60)
      Vdprev = Vd[ii]
      flush(stdout)
    end
  catch e
    u[:]    .= uprev[:]
    d[:]    .= dprev[:]

    error_msg = sprint(showerror, e)
    st        = sprint((io,v) -> show(io, "text/plain", v), 
                       stacktrace(catch_backtrace()))
    @warn "Trouble doing things:\n$(error_msg)\n$(st)" 
    println(" quitting." ); flush(stdout)
  end

  if bfailed
    println(" ### failed, exiting ")
  end

  saved_vars = Dict("sModelName"=>sModelName, "mat"=>mat, "dTolΔd"=>dTolΔd,
                    "maxiter"=>maxiter, "isave"=>isave, "nSteps"=>nSteps,
                    "Δy"=>Δy, "allus"=>allus, "Rf"=>Rf, "Δv"=>Δv,
                    "ϕmax"=>ϕmax, "sPostFix"=>sPostFix,
                    "bwithhist"=>bwithhist, "Vd"=>Vd,
                   )

  let sFileName = joinpath(sJLD2Path,sFileName*".jld2")
    println("\n saving binary data to: ", sFileName, " ..."); flush(stdout)
    FileIO.save(sFileName, saved_vars)
  end

  let item = writeVTKstate(spvdFileName, nodes, elems, u, 
                           elemsprop=get_ϵT(elems, u),
                           nodesprop=Dict("d"=>d, "u"=>u, "Fi"=>ru), 
                           bdef=bdef, bcenter=false)
    println("\n last state written to: ", item)
  end

  @printf("\n\nall done in %s\n", secs2hms((Base.time_ns()-tstart)/1e9))
  @show now()

  return saved_vars
end

"""
    solve_periodic_bc(sModelName="cm_quad1x1r500L1196lc0050lcf0750"; kwargs...) -> Dict

Solve a phase-field fracture problem on a periodic representative volume
element (RVE) of a two-phase (fiber/matrix) composite, under a
macroscopically prescribed strain history `ϵM0`. Periodicity is enforced
through explicit constraint matrices (`B0`, `Bϵ`, `B0d`, built by
[`make_B_matrices`](@ref) from paired boundary node sets `"left"`/`"right"`,
`"bottom"`/`"top"`, and, in 3D, `"front"`/`"back"`) rather than through
Lagrange-multiplier constraint equations, allowing the reduced,
periodicity-consistent linear systems to be factorized directly (via
`Pardiso`) and reused across Newton iterations for performance.

# Arguments
- `sModelName = "cm_quad1x1r500L1196lc0050lcf0750"`: base name of the
  Abaqus `.inp` RVE mesh, searched for in `sMeshPath`. The mesh must define
  a `"matrix"` node set and, if fibers are present, a `"fibers"` element
  set, plus the periodic boundary node-set pairs described above.

# Keyword Arguments
- `sMeshPath = joinpath(pkgdir(@__MODULE__), "meshes")`: mesh directory.
- `sJLD2Path = "jld2_files"`, `svtkPath = "vtk_files"`, `sPath = pwd()`:
  output locations; VTK/JLD2 paths are relative to `sPath`.
- `θ = 0.0`: rotation angle (radians) applied to the mesh geometry and to
  the periodicity vectors before assembling the constraint matrices —
  effectively rotates the loading direction relative to the microstructure
  without remeshing.
- `nSteps = 50`: number of load steps.
- `ϵM0 = [1, NaN, NaN]*1e-3`: target macroscopic strain vector at load
  factor `1`, in Voigt notation (`[ϵ11, ϵ22, ϵ12]` in 2D). Entries equal to
  `NaN` are left traction-free (unconstrained) rather than strain-prescribed.
- `sPostFix = "_e11NaN"`: output file suffix.
- `fiber_mat`, `matrix_mat`: `PhaseField` material models for the fiber and
  matrix phases respectively (see source for full defaults — both default
  to AT1 `Hooke2D` plane-stress materials with phase-appropriate `E`, `ν`,
  `l0`, and critical strain).
- `bwithhist = false`: history-field vs. direct-KKT damage update.
- `λV = 1.0`, `λT = 1e-1`: scale factors applied to the potential-energy
  and (numerically regularizing) mass-matrix contributions, respectively,
  when assembling the constrained tangent — `λT` stabilizes the reduced
  system without materially affecting the quasi-static solution.
- `dTol_ai = 1e-6`: tolerance used when pairing boundary nodes across
  periodic faces (nodes closer than this, after accounting for the
  periodicity vector, are treated as a matched pair).
- `dTolu = 1e-3`: convergence tolerance on the displacement residual.
- `dTold = 1e-5`: convergence tolerance on the damage sub-solver.
- `maxiter = 3`: maximum *global* (u–d coupled) iterations per load step.
- `maxiteru = 5`: maximum iterations for the reduced-system displacement
  solve within each global iteration.
- `maxiterd = 20`: maximum iterations for the damage sub-solver.
- `iterupdt = maxiteru`: number of consecutive iterations at the
  displacement-solve iteration cap before the constrained tangent
  (`B1KtB1ff`) is refactorized — trades some staleness in the tangent for
  avoiding an expensive Pardiso re-factorization every global iteration.
- `maxδdTol = 5e-2`: convergence tolerance on the maximum nodal change in
  damage `d` between global iterations within a load step.
- `bdef = false`: export the deformed configuration to ParaView.
- `bwritemodel = false`: include the full mesh/constraint-matrix data
  (`elems`, `nodes`, `B0`, `Bϵ`, node/element sets) in the saved `.jld2`
  file — off by default since this can be large.
- `ikeep = 5`, `nwritevtu = ikeep`: snapshot/VTK-output interval, in steps.
- `Nupdtmin = 3`: minimum number of steps between forced tangent updates.
- `maxnorm = 1e2`: abnormal-residual threshold; the run is aborted and the
  last good state restored if the displacement or damage residual exceeds
  this at any point.
- `bmake_χM = false`: additionally compute the homogenized (macroscopic)
  stiffness tensor `χM` of the *undamaged* RVE at the start of the run
  (expensive; off by default).
- `becho = true`: print per-iteration solver diagnostics.

# Returns
A `Dict` including:
- `"ϵM"`, `"σM"`: macroscopic strain and (volume-averaged) stress history,
  each an `length(ϵM0) × nSteps` array — the homogenized stress-strain
  response of the RVE.
- `"Vd"`, `"Vddot"`: damaged-volume fraction and its increment, per step.
- `"χM"`, `"Cm"`: homogenized stiffness (if `bmake_χM=true`, else `NaN`s)
  and its inverse (compliance).
- `"Vol"`, `"Vol_fibers"`, `"Vol_matrix"`: reference volumes of the RVE and
  each constituent phase.
- `"allus"`: `(u, d, step)` snapshots saved every `ikeep` steps.
- `"fiber_mat"`, `"matrix_mat"`, `"ϵM0"`, `"θ"`, and the various tolerance/
  iteration-count parameters above, saved for reproducibility.

# Example
```julia
results = solve_periodic_bc(
    ϵM0 = [1, NaN, NaN]*2.5e-3, θ = 0.0,
    fiber_mat  = Materials.PhaseField{...}(0.5, Gcf, hooke_fiber,  1),
    matrix_mat = Materials.PhaseField{...}(0.5, Gcm, hooke_matrix, 1),
    bwithhist = false, sPostFix = "nhAT1e11NaNNaNt")
```
"""
function periodicbc_solver(sModelName = "cm_tria1x1r500L1196lc0075lcf0300";
                           sMeshPath  = joinpath(pkgdir(@__MODULE__), "meshes"),
                           sJLD2Path  = "jld2_files",
                           svtkPath   = "vtk_files",
                           θ          = 0.0,
                           nSteps     = 50,
                           ϵM0        = [1, NaN, NaN]*1e-3,
                           sPostFix   = "_e11NaN",
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
                           bwithhist  = false,    # whether to use a history field to update d
                           λV         = 1.0,      # scale factor for potential energy
                           λT         = 1.e-1,    # scale factor for kinetic energy
                           dTol_ai    = 1e-6,     # tolerance in node pairing
                           dTolu      = 1e-3,     # convergence tolerance on u 
                           dTold      = 1e-5,     # convergence tolerance on d 
                           maxiter    = 3,        # maximum global iterations
                           maxiteru   = 5,        # maximum iterations on u
                           maxiterd   = 20,       # maximum iterations on d
                           iterupdt   = maxiteru, # number of failed iteration before updating B1KtB1ff
                           maxδdTol   = 5e-2,     # tolerance in maximum damage field change for convergence check
                           sPath      = pwd(),    # path for writing output files
                           bdef       = false,    # export deformed shape in paraview
                           bwritemodel= false,    # save the model in the binary data file
                           ikeep      = 5,        # keep config every n steps
                           nwritevtu  = ikeep,    # write vtu file after n iterations 
                           Nupdtmin   = 3,        # minimum number of steps between matrix update
                           maxnorm    = 1e2,      # abnormal residual produces termination
                           bmake_χM   = false,    # whether to compute macroscopic stiffness matrix
                           becho      = true,
                          )

  @show sMeshPath
  println("Model name: ", sModelName)

  # Ensure output directories exist
  mkpath(svtkPath)
  mkpath(sJLD2Path)

  println()
  @show t_start = Dates.now()
  @show nDoFs = 2
  @show ϵM0
  @show θ

  # Local function to compute macroscopic stiffness matrix
  function make_χM(maxiterχM=5, dTolχM=1e-5) 
    # Initialize stiffness matrix and solver
    Ktu = makeKt(Φ, elems, zeros(nDoFs,nNodes), zeros(1,nNodes))
    ps  = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    fix_iparm!(ps, :N)

    # Compute constrained stiffness matrix
    B0tKtuB0 = get_matrix(ps, transpose(B0)*(λT*MM+λV*Ktu)*B0, :N)

    # Initialize variables for iterative solution
    u0ϵ  = zeros(size(B0,2), size(Bϵ,2))
    updt = zeros(size(B0,2), size(Bϵ,2))
    Dϵ   = B0*u0ϵ + Bϵ
    res  = transpose(B0)*Ktu*Dϵ
    @printf("ii: %2i, maximum(abs.(res)): % .3e\n", 0, maximum(abs.(res))...)

    # Set up solver phases
    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, B0tKtuB0, res)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, B0tKtuB0, res)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)

    # Iterative solution
    for ii=1:maxiterχM
      pardiso(ps, updt, B0tKtuB0, res)
      u0ϵ -= updt
      Dϵ   = B0*u0ϵ + Bϵ
      res  = transpose(B0)*Ktu*Dϵ
      @printf("ii: %2i, maximum(abs.(res)): % .3e\n", ii, maximum(abs.(res))...)
      flush(stdout)
      maximum(abs.(updt)) ≤ dTolχM && break    
    end

    # Clean up solver
    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)

    # Return macroscopic stiffness matrix
    Matrix(transpose(Dϵ)*Ktu*Dϵ)/Vol
  end

  # Display parallel execution info
  println("\n running with ", Threads.nthreads(), " thread(s)"); flush(stdout)
  sFileName = sModelName*sPostFix
  println("\n sFileName: ", sFileName, "\n"); flush(stdout)

  # Initialize ParaView collection if needed
  spvdFileName  = joinpath(sPath, svtkPath, sFileName)
  if !isnan(nwritevtu)
    paraview_collection(spvdFileName) do pvd; end
  end

  # Create model and B-matrices for periodic boundary conditions
  nodes, elems, node_sets, 
  elem_sets, B0, Bϵ, B0d, ai  = make_the_2Dmodel(joinpath(sMeshPath, sModelName),
                                                 fiber_mat, matrix_mat,
                                                 θ, dTol_ai)

  # Display material properties
  println("fiber_mat:")
  dump(fiber_mat)
  println("\nmatrix_mat:")
  dump(matrix_mat)
  println()

  # Compute critical energy values
  ϕcf  = fiber_mat.Gc/4fiber_mat.l0     # critical energy for fibers
  ϕcm  = matrix_mat.Gc/4matrix_mat.l0   # critical energy for matrix

  # Count elements and compute volumes
  nElems_f    = length(elems[1])
  nElems_m    = length(elems[2])
  Vols_fibers = map(elem->elem.V, elems[1])
  Vols_matrix = map(elem->elem.V, elems[2])
  Vol_fibers  = sum(elem->elem.V, elems[1])
  Vol_matrix  = sum(elem->elem.V, elems[2])
  Vol         = Vol_matrix + Vol_fibers
  elems       = vcat(elems[1], elems[2])  # Combine fiber and matrix elements
  B1          = hcat(B0, Bϵ)  # Combined constraint matrix
  nElems      = length(elems)
  nNodes      = length(nodes)
  nDoFstot    = nNodes*nDoFs

  # Display volume information
  @show Vol_matrix
  @show Vol_fibers
  @show Vol
  @show nElems
  @show nElems_f
  @show nElems_m
  @show nNodes
  @show nDoFstot

  # Prepare arrays for constrained degrees of freedom
  nDoFstot, nDoFs0 = size(B0)
  bϵfree      = isnan.(ϵM0)  # Identify free strain components
  iϵcnst      = findall(.!bϵfree)
  bfree       = vcat(trues(nDoFs0), bϵfree)
  ifree       = findall(bfree)
  icnst       = findall(.!bfree)
  nϵ          = length(ϵM0)

  # Initialize state variables
  u           = zeros(nDoFs, nNodes)
  u1          = 1e-08randn(nDoFs0+nϵ)  # Small random initial values
  d           = zeros(1, nNodes)
  d0          = zeros(size(B0d,2))
  d[:]        = B0d*d0  # Project initial damage field

  # Initialize history field if needed
  ϕmax        = if bwithhist
    [[(0., 0.) for ii in 1:length(elem.wgt)] for elem in elems]
  else
    nothing
  end
  @show typeof(ϕmax)

  # Create mass matrix
  _, _, MM    = getT(elems, u)

  # Create damage constraint matrix
  B0dtB0d     = let 
    ps      = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.REAL_SYM_POSDEF)
    fix_iparm!(ps, :N)
    B0dtB0d = get_matrix(ps, transpose(B0d)*B0d, :N)

    set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)

    B0dtB0d
  end

  # Initialize energy functionals
  println("\n making hessians ... "); flush(stdout)
  @time Φ     = [getϕ(elem, adiff.D2(u[:,elem.nodes]), d[elem.nodes]) for elem in elems]   

  # Compute macroscopic stiffness matrix if requested
  if bmake_χM
    println("\n calculating χM: "); flush(stdout)
    @time χM = make_χM() 
    print("\n χM = ")
    display(round.(χM, digits=2))
    flush(stdout)
    Cm = inv(χM)
  else
    Cm = χM = fill(NaN, length(ϵM0), length(ϵM0))
  end

  # Initialize damage stiffness matrix
  println("doing makeϕrKt_d ... ")
  @time begin
    _, _, Ktd    = makeϕrKt_d(elems, zeros(nDoFs,nNodes), zeros(1,nNodes))
    B0dtKtdB0d   = transpose(B0d)*Ktd*B0d
  end
  println(" ... done"); flush(stdout)

  # Main simulation loop variables
  dprev     = copy(d)
  uprev     = copy(u)
  allus     = []  # Storage for saved configurations
  Vdprev    = getVd(elems, d)/Vol
  cntupdt   = 0  # Counter for matrix updates
  ru        = zeros(nDoFs*nNodes)  # Residual vector
  normruii  = normrdii = 0.0  # Norms of residuals

  # Initialize result arrays
  ϵM        = fill(NaN, length(ϵM0), nSteps)
  σM        = fill(NaN, length(ϵM0), nSteps)
  Vd        = fill(NaN, nSteps) 
  Vddot     = fill(NaN, nSteps) 

  # Function to update the constrained stiffness matrix
  function make_B1KtB1ff(ps, u, d)
    Ktu      = makeKt(Φ, elems, u, d)
    res      = transpose(B1[:,ifree])*Ktu*u[:]
    B1KtB1ff = get_matrix(ps, transpose(B1[:,ifree])*(λT*MM+λV*Ktu)*B1[:,ifree], :N)

    # Set up solver for the updated matrix
    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, B1KtB1ff, res)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, B1KtB1ff, res)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)

    return B1KtB1ff
  end

  # Initialize solver for displacement updates
  psu     = MKLPardisoSolver()
  set_matrixtype!(psu, Pardiso.REAL_SYM_POSDEF)
  fix_iparm!(psu, :N)

  # Create initial constrained stiffness matrix
  println("\t==== finding B1KtB1ff ... "); flush(stdout)
  @time B1KtB1ff = make_B1KtB1ff(psu, u, d) 
  println("\t\t ... done "); flush(stdout)

  # Start main simulation loop
  println("\n starting ... "); flush(stdout)
  t0       = Base.time_ns()
  totiters = 0
  iStep    = 1
  LF       = range(0,1,length=nSteps)  # Load factors
  try
    while iStep < nSteps
      printstyled("\nstarting step $iStep/$nSteps, LF: ", 
                  round(LF[iStep],sigdigits=3), "\n", color=:blue)

      # Write VTK output if needed
      if iStep % nwritevtu == 0
        paraview_collection(spvdFileName; append=true) do pvd
          println(" state written to:\n ",
                  writeVTKstate(spvdFileName, nodes, elems, u, 
                                pvd=pvd, 
                                elemsprop=get_ϵT(elems, u),
                                nodesprop=Dict("d"=>d), 
                                bdef=false, bcenter=false,ii=iStep))
        end
      end

      # Step initialization
      stept0     = Base.time_ns()
      u1[icnst]  = ϵM0[iϵcnst]*LF[iStep]  # Apply boundary conditions
      u[:]       = B1*u1  # Update displacement field
      σMold      = transpose(Bϵ)*ru/Vol  # Previous stress

      # Display current strain state
      # println(" ϵM    = [", join([@sprintf("%+3.2e, ", x) for x in u1[icnst]]), "]")
      println(" ϵM    = [", join([@sprintf("%+3.2e, ", x) for x in ϵM0*LF[iStep]]), "]")

      # Iterative solution for current step
      iter, maxδd = 0, Inf
      while iter ≤ maxiter && maxδd > maxδdTol
        t1       = Base.time_ns()

        # Update displacements
        normruii, iteru  = update_u!(psu, elems, Φ, B1KtB1ff, ifree, 
                                     ru, u, u1, d, 
                                     B1, dTolu, maxiteru)

        # Compute stress change
        σMnew    = transpose(Bϵ)*ru/Vol
        ΔσM      = σMnew-σMold
        println(" ΔσM   = [", join([@sprintf("%+3.2e, ", x) for x in ΔσM[iϵcnst]]), "]")
        σMold[:] = σMnew[:]

        # Update damage field
        dold      = deepcopy(d)
        update_d!(elems, u, d, B0d, d0, ϕmax, maxiter=maxiterd, 
                  becho=becho, dTol=dTold)
        maxδd     = maximum(abs.(d-dold))

        # Display iteration info
        iterutime = (Base.time_ns()-t1)/1e9
        @printf("\n step-iter:%-2i, |ru|:%.3e, maxδd:%.3e, in %.2f sec\n\n", 
                iter, normruii, maxδd, iterutime )
        totiters += 1

        # Update stiffness matrix if needed
        if iteru≥iterupdt 
          if cntupdt≥Nupdtmin
            println("\t==== updating B1KtB1ff ... ")
            @time begin
              B1KtB1ff = make_B1KtB1ff(psu, u, d) 
              GC.gc()  # Garbage collect after update
            end
            println("\t\t ... done "); flush(stdout) 
            cntupdt   = 0
          else
            @printf("\t---- updating B1KtB1ff in %3i steps \n", 
                    Nupdtmin-cntupdt); flush(stdout)
          end
        end

        # Check for abnormal residuals
        if (normruii>maxnorm) || (normrdii>maxnorm) 
          println("abnormal residual, quitting.")
          u[:]    .= uprev[:]
          d[:]    .= dprev[:]
          break
        end

        iter     +=1
        flush(stdout)
      end

      # Handle non-convergence
      if iter == maxiter
        @printf("\t#### not converged, continuing ####\n"); flush(stdout)
      end

      # Check for NaN values
      any(isnan.(u[:])) && error("NaN in u detected")
      any(isnan.(d[:])) && error("NaN in d detected")

      # Check for abnormal residuals again
      if (normruii>maxnorm) || (normrdii>maxnorm) 
        break
      end

      # Record final stress state
      σMnew       = transpose(Bϵ)*ru/Vol
      # println(" σM    = [", join([@sprintf("%+3.2e, ", x) for x in σMnew[iϵcnst]]), "]")
      println(" σM    = [", join([@sprintf("%+3.2e, ", x) for x in σMnew]), "]")

      # Update state variables
      Δd          = d - dprev
      dprev[:]    = d[:]
      uprev[:]    = u[:]

      # Update counters and record results
      cntupdt    += 1
      Vd[iStep]      = getVd(elems, d)/Vol
      Vddot[iStep]   = (Vd[iStep]-Vdprev)
      Vdprev      = Vd[iStep]
      σM[:,iStep]    = σMnew
      ϵM[:,iStep]    = u1[nDoFs0+1:end]

      iStep += 1

      # Save configuration if needed
      if iStep%ikeep==0
        push!(allus, (copy(u), copy(d), iStep))
      end

      # Display timing information
      tottime  = (Base.time_ns()-t0)/1e9
      stptime  = (Base.time_ns()-stept0)/1e9

      @printf(" Δd: %0.4f … %0.4f → d: %0.4f … %0.4f\n", extrema(Δd)..., extrema(d)...)
      println("Step $iStep/$nSteps completed in ", round(stptime / 60, digits = 2), " mins., ",
              "Elapsed time ", secs2hms(tottime), ", ",
              "ETA ", secs2hms((nSteps - iStep) * stptime), "\n")
    end

    # Display final timing information
    @printf("\n all done in %5.1f mins, with %-4i steps, %-4i total iterations\n", 
            (Base.time_ns()-t0)/1e9/60, iStep, totiters); flush(stdout)

  catch e
    # Handle errors and restore previous state
    u[:]    .= uprev[:]
    d[:]    .= dprev[:]

    ϵM       = ϵM[:,1:iStep]
    σM       = σM[:,1:iStep] 
    Vd       = Vd[1:iStep]
    Vddot    = Vddot[1:iStep]

    error_msg = sprint(showerror, e)
    flush(stdout)
    st        = sprint((io, v) -> show(io, "text/plain", v),
                       stacktrace(catch_backtrace()))
    @warn "Simulation error at step $iStep:\n$(error_msg)\n$(st)"
    println("Quitting.")
    flush(stdout)
  end

  # Clean up solver
  set_phase!(psu, Pardiso.RELEASE_ALL)
  pardiso(psu)

  # Write final state to ParaView
  println(" state written to:\n ", 
          writeVTKstate(spvdFileName, nodes, elems, u, 
                        elemsprop=get_ϵT(elems, u),
                        nodesprop=Dict("d"=>d), 
                        bdef=true, bcenter=false))    
  flush(stdout)

  # Prepare results dictionary
  t_end      = Dates.now()
  saved_vars = Dict("sModelName"=>sModelName, "sPostFix"=>sPostFix,
                    "sFileName"=>sFileName,   "sPath"=>sPath,
                    "maxiter"=>maxiter,       "maxiteru"=>"maxiteru", 
                    "dTolu"=>dTolu, "dTold"=>dTold,
                    "maxδdTol"=>maxδdTol,
                    "λV"=>λV,     "λT"=>λT,
                    "Vol"=>Vol,   "Vol_fibers"=>Vol_fibers, "Vol_matrix"=>Vol_matrix, 
                    "ai"=>ai,     "nDoFs"=>nDoFs,   "ϵM0"=>ϵM0, 
                    "θ"=>θ, "bwithhist"=>bwithhist,
                    "dTol_ai"=>dTol_ai,       "bϵfree"=>bϵfree,
                    "ϵM"=>ϵM,     "σM"=>σM,   "Vd"=>Vd, 
                    "χM"=>χM,     "Cm"=>Cm,
                    "fiber_mat"=>fiber_mat,     "matrix_mat"=>matrix_mat,
                    "LF"=>LF, "nSteps"=>nSteps,
                    "t_start"=>t_start, "t_end"=>t_end, "Vddot"=>Vddot,
                    "ϕmax"=>ϕmax, "sPostFix"=>sPostFix,
                    "allus"=>allus)

  # Include model data if requested
  if bwritemodel
    merge!(saved_vars, Dict("elems"=>elems, "nodes"=>nodes, "B0"=>B0, "Bϵ"=>Bϵ, 
                            "node_sets"=>node_sets, "elem_sets"=>elem_sets))
  end

  # Save results to JLD2 file
  @time let sFileName = joinpath(sPath,sJLD2Path,sFileName*".jld2")
    println("\n saving binary data to \n\t", sFileName, " ..."); flush(stdout)
    FileIO.save(sFileName, saved_vars)
  end

  println(" exiting. ")
  @show Dates.now()
  saved_vars
end

"""
    update_d!(elems, u, d, ϕmax; kwargs...)

Update phase-field variable d using KKT conditions or history-based method.

# Arguments
- `elems`: Element collection
- `u`: Displacement field
- `d`: Phase-field variable
- `ϕmax`: History variable (nothing for KKT method)

# Keyword Arguments
- `dTol`: Convergence tolerance
- `maxiter`: Maximum iterations
- `becho`: Enable verbose output

# Returns
Residual vector
"""
function update_d!(elems, u, d, ϕmax::Nothing;
                   dTol=1e-5, maxiter=10, becho=true)
  becho && print("\n--- doing update_d with KKT alt")

  _,rd,Ktd = makeϕrKt_d(elems, u, d)

  return do_dupdate!(d, rd, Ktd, dTol, maxiter, becho)
end

function update_d!(elems, u, d, ϕmax::Vector;
                   dTol=1e-6, maxiter=10,becho=true)
  becho && print("--- doing update_d, with history \n")

  _,res,Ktd = makeϕrKt_d(elems, u, d, ϕmax)
  becho &&  @printf("\t extrema(res)       : % .3e, % .3e\n", extrema(res)...)

  if any(res.<0)
    updt           = lu(Ktd)\-res
    updt[updt.<0] .= 0 
    d[:]          += updt 
    clamp!(d, 0, 1)

    if becho
      @printf("\t extrema(Δd)        : % .3e, % .3e\n", extrema(updt)...)
    end
  end
  res
end

function update_d!(elems, u, d, B0d, d0, ϕmax::Nothing; 
                   dTol=1e-5, maxiter=10, becho=true)
  print("\n doing update_d with KKT, ")

  _,rd,Ktd  = makeϕrKt_d(elems, u, d)

  Ktd = transpose(B0d)*Ktd*B0d
  rd  = transpose(B0d)*rd[:]

  r     = do_dupdate!(d0, rd, Ktd, dTol, maxiter, becho)
  d[:] .= B0d*d0
  return r
end

function update_d!(elems, u, d, B0d, d0, ϕmax::Vector; 
                   dTol=1e-5, maxiter=10, becho=true)
  becho && print("--- doing update_d, with history \n")
  extremau  = (NaN, NaN)

  _,res,Ktd = makeϕrKt_d(elems, u, d, ϕmax)
  Ktd       = transpose(B0d)*Ktd*B0d
  res       = transpose(B0d)*res[:]

  if any(res.<0)
    updt      = lu(Ktd)\-res
    updt[updt.<0] .= 0 
    d0[:]    += updt 
    clamp!(d0, 0, 1)
    extremau=extrema(updt)
  end
  dprev     = copy(d)
  d[:]      = B0d*d0

  extremaΔd = extrema(d-dprev)
  if becho
    @printf("\t extrema(res)       : % .3e, % .3e\n", extrema(res)...)
    @printf("\t extrema(updt)      : % .3e, % .3e\n", extremau...)
    @printf("\t extrema(Δd)        : % .3e, % .3e\n", extremaΔd...)
    @printf("\t extrema(d)         : % .3e, % .3e\n", extrema(d)...)
  end
  return extrema(res), extremau
end

"""
    update_u!(elems, u, d, ifree, ru; kwargs...)

Update displacement field using Newton-Raphson method.

# Arguments
- `elems`: Element collection
- `u`: Displacement field
- `d`: Phase-field variable
- `ifree`: Free DOF indices
- `ru`: Reaction force vector

# Keyword Arguments
- `dTolMax`: Maximum allowed displacement increment
- `becho`: Enable verbose output

# Returns
Boolean indicating convergence success
"""
function update_u!(elems, u, d, ifree, ru; dTolMax=1e2, becho=true)
  becho && print("--- doing update_u, \n")

  _,r,Kt  = makeϕrKt(elems, u, d)
  updt    = lu(Kt[ifree,ifree])\r[ifree]
  # updt    = qr(Kt[ifree,ifree])\r[ifree]
  Δu      = extrema(updt)

  if all(abs.(Δu).<dTolMax) 
    u[ifree] .-= updt
    ru[:]    .=  Kt*u[:]
    if becho
      @printf("\t extrema(Δu)        : % .3e, % .3e\n", Δu...)
      @printf("\t extrema(u)         : % .3e, % .3e\n", extrema(u)...)
      @printf("\t extrema(ru[ifree]) : % .3e, % .3e\n", extrema(ru[ifree])...)
    end
    return true
  else
    return false
  end

end

"""
    update_u!(elems, eqns, u, d, ru; kwargs...)

Update displacement field with constraint equations.

# Arguments
- `elems`: Element collection
- `eqns`: Constraint equations
- `u`: Displacement field
- `d`: Phase-field variable
- `ru`: Reaction force vector

# Keyword Arguments
- `dTolMax`: Maximum allowed displacement increment
- `dTol`: Residual tolerance
- `maxiter`: Maximum iterations
- `becho`: Enable verbose output

# Returns
Boolean indicating convergence success
"""
function update_u!(elems, eqns::Vector{ConstEq}, u, d, ru;
                   dTolMax  = 1e3,
                   dTol     = 1e-6,
                   maxiter  = 10,
                   becho    = false)

  becho && print(" updating u, ")

  uold  = copy(u)
  nEqs  = length(eqns)
  nDoFs = length(u) + nEqs
  iius  = 1:length(u)
  ieqs  = length(u) .+ (1:nEqs)

  λ     = zeros(nEqs)
  H     = spzeros(nDoFs, nDoFs)

  _,ru[:],ktu = makeϕrKt(elems, u, d)
  iter  = 0
  while true

    vEqs,rEqs,KEqs = makeϕrKt(eqns, u, λ)
    res            = vcat(ru[:]+rEqs*λ, vEqs)
    extremares     = extrema(res)
    if all(abs.(extremares).<dTol) || iter>maxiter
      becho && println(" done in $iter iterations")
      becho && @printf("\t extrema(res)       : % .3e, % .3e\n", extremares...)
      break
    end

    H[iius,iius]  = ktu + KEqs
    H[iius,ieqs]  = rEqs
    H[ieqs,iius]  = transpose(rEqs)
    H            += spdiagm(0=>dTol*randn(nDoFs))

    updt  = lu(H)\res;
    u[:] -= updt[iius]
    λ    -= updt[ieqs]
    ru[:] = ktu*u[:]

    if becho 
      @printf("\t extrema(updt[iius]): % .3e, % .3e\n", extrema(updt[iius])...)
      @printf("\t extrema(updt[ieqs]): % .3e, % .3e\n", extrema(updt[ieqs])...)
    end
    iter += 1
  end

  Δu = u-uold
  if all(abs.(Δu).<dTolMax) 
    if becho 
      @printf("\t extrema(Δu)        : % .3e, % .3e\n", extrema(Δu)...)
      @printf("\t extrema(u)         : % .3e, % .3e\n", extrema(u)...)
    end
    return true
  else
    return false
  end

end

function update_u!(ps, elems, Φ, B1KtB1ff, ifree, ru, u, u1, d, 
                   B1, dTolu, maxiter)

  iter    = 0
  N       = length(ifree)
  normres = normupdt= NaN
  Ktu     = makeKt(Φ, elems, u, d)
  ru[:]   = Ktu*u[:]
  res     = transpose(B1[:,ifree])*Ktu*u[:]
  updt    = similar(res)

  println("\tu-iter\tnormresu\tnormupdtu")
  while true 
    normres = maximum(abs.(res))
    if iter≥maxiter || normres≤dTolu
      @printf("\t%i\t%.2e\t%.2e\n", 
              iter, normres, normupdt)
      break
    end

    pardiso(ps, updt, B1KtB1ff, res)
    u1[ifree] -= updt
    u[:]       = B1*u1
    normupdt   = √(updt⋅updt)
    ru[:]      = Ktu*u[:]
    res        = transpose(B1[:,ifree])*ru
    @printf("\t%i\t%.2e\t%.2e\n", 
            iter, normres, normupdt)
    iter      += 1
  end
  flush(stdout)

  normres, iter
end

end

