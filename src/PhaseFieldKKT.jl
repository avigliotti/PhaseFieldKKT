module PhaseFieldKKT

VER = "v0.1.0"

using  AD4SM

export AD4SM, Solvers, Materials, Elements

include("helper_funcs.jl")
include("phasefield_solvers.jl")

using .PhaseFieldSolvers

export PhaseFieldSolvers

end
