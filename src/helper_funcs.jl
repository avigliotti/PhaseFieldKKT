
module HelperFuncs

using WriteVTK, Printf, LinearAlgebra, SparseArrays
using AbaqusReader, Logging, FileIO, Dates

using AD4SM
using  .Solvers, .Materials, .Elements

export get_ϵT, writeVTKstate, secs2hms, do_dupdate!
export make_the_2Dmodel, getT, get_cg, makeKt

"""
    secs2hms(secs)

Convert seconds to HH:MM:SS formatted string.

# Arguments
- `secs`: Time in seconds

# Returns
Formatted time string
"""
function secs2hms(secs)
  h, r = divrem(secs, 3600)
  m, s = divrem(r, 60)
  @sprintf("%02i:%02i:%02i",h, m, s)
end

"""
    get_ϵT(elems::Vector{<:CPElem{D,<:PhaseField} where D}, u::Array{T,2} where T)

Calculate the maximum principal strain difference (ϵT) for each element in a collection.

# Arguments
- `elems`: Vector of phase-field continuum elements with P integration points
- `u`: Displacement field array of size (nDoFs × nNodes)

# Returns
- Dictionary mapping the string "epsilon_T" to an array of ϵT values for each element

# Description
This function computes the maximum difference between principal strains (ϵT) for each element,
which represents the maximum shear strain. It averages the deformation gradient over integration
points, computes the right Cauchy-Green deformation tensor, extracts principal stretches via SVD,
and returns the difference between maximum and minimum principal stretches.
"""
function get_ϵT(elems::Vector{<:CPElem}, u::Matrix{T} where T)
  Dict("\$\\epsilon_T\$"=>[get_ϵT(elem, u[:,elem.nodes]) for elem in elems])
end

"""
    get_ϵT(elem::CPElem{D,P,<:PhaseField} where D, u0::Array{N,2} where N) where P

Calculate the maximum principal strain difference (ϵT) for a single element.

# Arguments
- `elem`: A phase-field continuum element with P integration points
- `u0`: Displacement field for the element's nodes of size (nDoFs × nNodes)

# Returns
- ϵT value representing the maximum difference between principal strains

# Description
Computes the maximum shear strain (ϵT) for a single element by:
1. Calculating deformation gradient at each integration point
2. Computing the right Cauchy-Green deformation tensor
3. Extracting principal stretches via singular value decomposition
4. Averaging stretches over integration points
5. Returning the difference between maximum and minimum principal stretches
"""
function get_ϵT(elem::CPElem{D,P,M} where {D,M},
                u0::Matrix{T} where T) where P
  wgt  = elem.wgt
  F    = [Elements.getF(elem, u0, ii)                 for ii=1:P]
  ϵn   = sum([wgt[ii]*(transpose(F[ii])+F[ii]-2I)/2   for ii=1:P])/elem.V
  C    = [transpose(F[ii])F[ii]                       for ii=1:P]
  λp   = sum([wgt[ii]*sqrt.(svdvals(C[ii]))           for ii=1:P])/elem.V
  ϵT   = maximum(λp) - minimum(λp)
end

"""
    writeVTKstate(sFileName, nodes, elems, u; pvd, elemsprop, nodesprop, ii, r0, bdef, bcenter)

Write the current simulation state to a VTK file for visualization.

# Arguments
- `sFileName`: Base filename for output (without extension)
- `nodes`: Array of node coordinates
- `elems`: Vector of finite elements
- `u`: Displacement field (optional)

# Keyword Arguments
- `pvd`: ParaView data collection object for time series (optional)
- `elemsprop`: Dictionary of element properties to output (optional)
- `nodesprop`: Dictionary of node properties to output (optional)
- `ii`: Time step index for series output (default: 0)
- `r0`: Reference coordinates offset (default: zeros)
- `bdef`: Whether to write deformed configuration (default: false)
- `bcenter`: Whether to center displacements (default: true)

# Returns
- Filename of the created VTK file

# Description
This function writes the current simulation state to a VTK file for visualization in ParaView.
It handles both undeformed and deformed configurations, supports element and node properties,
and can be used for time series output when provided with a ParaView data collection object.
"""
function writeVTKstate(sFileName,
                       nodes::Array{Array{D,1},1} where D<:Number, 
                       elems::Vector{E},
                       u          = []; 
                       pvd        = nothing, 
                       elemsprop  = Dict{String,Any}(),
                       nodesprop  = Dict{String,Any}(),
                       ii         = 0, 
                       r0         = zeros(size(nodes[1])),
                       bdef       = false,
                       bcenter    = true) where E<:CPElem

  # Determine cell type based on element properties
  N = length(elems[1].nodes)
  cellType = determine_cell_type(E, N)

  # Initialize node coordinates
  nNodes = length(nodes)
  nDoFs = length(nodes[1])
  points = calculate_points(nodes, u, r0, bdef, bcenter, nNodes, nDoFs)

  # Adjust file name if part of a series
  if !isnothing(pvd)
    sFileName = @sprintf("%s_%04i", sFileName, ii)
  end

  # Create mesh cells and write to VTK
  cells = [WriteVTK.MeshCell(cellType, elem.nodes) for elem in elems]
  WriteVTK.vtk_grid(sFileName, points, cells) do vtkobj
    write_vtk_data(vtkobj, elemsprop, nodesprop)
    if !isnothing(pvd)
      pvd[ii] = vtkobj
    end
  end

  return sFileName
end

"""
    determine_cell_type(E, N)

Determine the VTK cell type based on element type and number of nodes.

# Arguments
- `E`: Element type
- `N`: Number of nodes per element

# Returns
- VTKCellTypes value representing the appropriate cell type

# Description
Maps Julia element types to corresponding VTK cell types for proper visualization.
Supports tetrahedra, hexahedra, quadrilaterals, triangles, and wedge elements.
"""
function determine_cell_type(E, N)
  if N == 4 && E <: C3D
    return VTKCellTypes.VTK_TETRA
  elseif N == 8 && E <: C3D
    return VTKCellTypes.VTK_HEXAHEDRON
  elseif N == 4 && E <: C2D
    return VTKCellTypes.VTK_QUAD
  elseif N == 3 && E <: C2D
    return VTKCellTypes.VTK_TRIANGLE
  elseif N == 6 && E <: C3D
    return VTKCellTypes.VTK_WEDGE
  else
    error("Element type ", E, " with ", N, " nodes not recognized")
  end
end

"""
    calculate_points(nodes, u, r0, bdef, bcenter, nNodes, nDoFs)

Calculate point coordinates for VTK output.

# Arguments
- `nodes`: Array of node coordinates
- `u`: Displacement field
- `r0`: Reference coordinates offset
- `bdef`: Whether to include deformation
- `bcenter`: Whether to center displacements
- `nNodes`: Number of nodes
- `nDoFs`: Number of degrees of freedom per node

# Returns
- Array of point coordinates for VTK output

# Description
Computes the final positions of nodes for visualization, optionally including
displacement effects and centering the configuration.
"""
function calculate_points(nodes, u, r0, bdef, bcenter, nNodes, nDoFs)
  points = zeros(nDoFs, nNodes)
  if bdef
    u_cg = bcenter ? sum([u[:, i] for i in 1:nNodes]) / nNodes : zeros(nDoFs)
    for i in 1:nNodes
      points[:, i] = nodes[i] + r0 + u[:, i] - u_cg
    end
  else
    for i in 1:nNodes
      points[:, i] = nodes[i] + r0
    end
  end
  return points
end

"""
    write_vtk_data(vtkobj, elemsprop, nodesprop)

Write element and node properties to a VTK object.

# Arguments
- `vtkobj`: VTK object to write data to
- `elemsprop`: Dictionary of element properties
- `nodesprop`: Dictionary of node properties

# Description
Adds element-based and node-based data to a VTK object for visualization.
Element properties are added as cell data, while node properties are added as point data.
"""
function write_vtk_data(vtkobj, elemsprop, nodesprop)
  if !isempty(elemsprop)
    for spropname in keys(elemsprop)
      WriteVTK.vtk_cell_data(vtkobj, elemsprop[spropname], spropname)
    end
  end
  if !isempty(nodesprop)
    for spropname in keys(nodesprop)
      WriteVTK.vtk_point_data(vtkobj, nodesprop[spropname], spropname)
    end
  end
end

function do_dupdate!(d::Array, rd::Vector, Kt::SparseMatrixCSC,
                     dTol=1e-5, maxiter=10, becho=true)

  nDoFs     = length(rd)
  Δd        = zeros(nDoFs)
  res       = zeros(nDoFs)

  bfree     = rd .< 0.0
  bfreeold  = copy(bfree)
  bcnst     = .!bfree

  iter      = 0
  while iter < maxiter
    Δd    .= 0
    if any(bfree)
      Δd[bfree]  = lu(Kt[bfree,bfree])\-rd[bfree]
    end
    res     .= Kt*Δd+rd

    if becho
      println("\n\titer         : ", iter)
      println("\tsum(bfree)   : ", sum(bfree))
      any(bfree) && println("\tΔd[bfree]  ∈ ", round.(extrema(Δd[bfree]), sigdigits=3),
                            ", all(Δd[bfree].≥ -dTol): ", all(Δd[bfree].≥ -dTol))
      any(bcnst) && println("\tres[bcnst] ∈ ", round.(extrema(res[bcnst]),sigdigits=3),
                            ", all(res[bcnst].≥ -dTol): ", all(res[bcnst].≥ -dTol))
    end

    bcond = all(Δd[bfree].≥ -dTol) && all(res[bcnst].≥ -dTol)
    if bcond
      d[bfree]   += Δd[bfree] 
      clamp!(d,0,1)
      becho && printstyled("\texited after $iter iterations \n", color=:blue)
      break
    else
      bfreeold    .= bfree
      bfree[bfree] = Δd[bfree]  .> 0
      bfree[bcnst] = res[bcnst] .< 0
      bcnst       .= .!bfree
    end

    if bfreeold == bfree
      becho && printstyled("\tstalled, exiting\n", color=:green)
      break
    end
    iter += 1
  end

  becho && iter≥maxiter && printstyled("\tfailed after $iter iterations \n", color=:red)

  return res
end

"""
    make_the_2Dmodel(sModelName, mat; bisAS=false)

Load 2D Abaqus mesh and create element objects.

# Arguments
- `sModelName`: Base filename without extension
- `mat`: Material model

# Keyword Arguments
- `bisAS`: Use axial-symmetric elements

# Returns
Tuple containing (nodes, elements, node_sets, element_sets)
"""
function make_the_2Dmodel(sModelName::String, mat::Material; bisAS=false)
  nDoFs = 2

  model = with_logger(Logging.NullLogger()) do
    AbaqusReader.abaqus_read_mesh(sModelName*".inp")
  end

  nNodes      = length(model["nodes"])
  nodes       = [ model["nodes"][ii][1:nDoFs] for ii in 1:nNodes ]
  elements    = model["elements"]
  node_sets   = model["node_sets"]
  elem_sets   = model["element_sets"]

  # @show length(node_sets["btm"]), length(node_sets["top"])

  elems = if bisAS
    [if length(elements[id])==4
       Elements.ASQuad(elements[id], nodes[elements[id]], mat=mat)
     elseif length(elements[id])==3
       Elements.ASTria(elements[id], nodes[elements[id]], mat=mat)
     else
       error("unknown element ", id, " with ", length(elements[id]), " nodes")
     end  for id in elem_sets["surf"] ]
  else
    [if length(elements[id])==4
       Elements.QuadP(elements[id], nodes[elements[id]], mat=mat)
     elseif length(elements[id])==3
       Elements.TriaP(elements[id], nodes[elements[id]], mat=mat)
     else
       error("unknown element ", id, " with ", length(elements[id]), " nodes")
     end  for id in elem_sets["surf"] ]
  end

  return nodes, elems, node_sets, elem_sets
end

function make_the_2Dmodel(sModelName::String, fiber_mat::Material, matrix_mat::Material, 
                          θ=0, dTol_ai=1e-6,
                          bfullint=true, bsmall=true)

  nDoFs    = 2
  # rotation matrix for the model, the sign is reversed in order for θ to point
  # to the direction of the applied strain
  M        = [cos(θ) sin(θ); -sin(θ) cos(θ)]    

  # load model from input file
  model = with_logger(Logging.NullLogger()) do
    AbaqusReader.abaqus_read_mesh(sModelName*".inp")
  end

  nNodes      = length(model["nodes"])
  nDoFstot    = nDoFs*nNodes
  nodes       = [ model["nodes"][ii][1:nDoFs] for ii in 1:nNodes ]
  elements    = model["elements"]
  node_sets   = model["node_sets"]
  elem_sets   = model["element_sets"]
  matrixnodes = node_sets["matrix"]
  hasfibers   = haskey(elem_sets, "fibers")

  (B0, B0d, Bϵ, (a1,a2)) = make_B_matrices(model, nDoFs=nDoFs, M=M)

  #
  #   constructs elements
  #
  @show hasfibers
  println("\n constructing elements ... "); flush(stdout)
  @time begin
    if hasfibers
      fibers_elems = [if length(elements[id])==4
                        Elements.QuadP(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                       mat=fiber_mat) 
                      else
                        Elements.TriaP(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                       mat=fiber_mat) 
                      end
                      for id in model["element_sets"]["fibers"]]
      Vol_fibers   =  sum([item.V for item in fibers_elems])
    else
      fibers_elems = []
      Vol_fibers   = 0.0
    end

    matrix_elems = [if length(elements[id])==4
                      Elements.QuadP(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                     mat=matrix_mat) 
                    else
                      Elements.TriaP(elements[id], [M*nodes[ii] for ii=elements[id]], 
                                     mat=matrix_mat) 
                    end
                    for id in model["element_sets"]["matrix"]]
    Vol_matrix  = sum([item.V for item in matrix_elems])

    elems = (fibers_elems, matrix_elems)
  end

  println(" ... done\n"); flush(stdout)

  return nodes, elems, node_sets, elem_sets, B0, Bϵ, B0d , (a1,a2)
end

"""
    get_cg(nodes)

Calculate center of gravity for a set of nodes.

# Arguments
- `nodes`: Collection of node coordinates

# Returns
Center of gravity coordinates
"""
function get_cg(nodes)
  cg = zero(nodes[1])
  for node in nodes
    cg += node
  end
  cg /= length(nodes)
end

function makeKt(Φ::Vector{<:adiff.D2{N,<:Any,T}}, elems::Vector{<:C2D}, u, d) where {N,T}

  @assert length(Φ)==length(elems) "length(Φ)!=length(elems)"

  NM     = length(u)
  oneii  = ones(Int,N)
  N1     = N*N
  Ntot   = length(elems)*N1
  II     = Vector{Int}(undef, Ntot)
  JJ     = Vector{Int}(undef, Ntot)
  Kt     = Vector{T}(undef, Ntot)
  idxs   = LinearIndices(u)

  for (ii,elem) in enumerate(elems)
      d_el    = getVd(elem, d[elem.nodes])/elem.V
      idxii   = idxs[:, elem.nodes][:]
      idd     = (ii-1)*N1+1:ii*N1
      II[idd] = idxii * transpose(oneii)
      JJ[idd] = oneii * transpose(idxii)
      Kt[idd] = (1-d_el)^2*adiff.hess(Φ[ii])
  end

  dropzeros(sparse(II,JJ,Kt,NM,NM))
end

"""
    make_B_matrices(model, nDoFs=3, M=I)

Generate B matrices for periodic boundary conditions in 2D or 3D.

# Arguments
- `model`: Model dictionary containing nodes and node sets
- `nDoFs`: Number of degrees of freedom (2 for 2D, 3 for 3D)
- `M`: Transformation matrix to apply to periodicity vectors

# Returns
Tuple containing:
- B0: Constraint matrix for periodic boundary conditions
- B0d: Reduced constraint matrix
- Bϵ: Strain-displacement matrix for periodic boundary conditions
- Tuple of transformed periodicity vectors
"""
function make_B_matrices(model::Dict; M::AbstractMatrix=I, nDoFs::Int=3)
  @assert nDoFs ∈ (2, 3) "nDoFs must be 2 or 3, got $nDoFs"

  nNodes    = length(model["nodes"])
  nodes     = [model["nodes"][ii][1:nDoFs] for ii in 1:nNodes]
  node_sets = model["node_sets"]

  # Validate pair sets
  pair_sets = if nDoFs==2 
    [("left", "right"), ("bottom", "top")] 
  else
    [("left", "right"), ("bottom", "top"), ("front", "back")]
  end

  foreach(pair_sets) do (set1, set2)
    @assert length(node_sets[set1]) == length(node_sets[set2]) 
    "Mismatch between $set1 ($(length(node_sets[set1]))) and $set2 ($(length(node_sets[set2]))) nodes"
  end

  # Find pairs for all relevant directions
  pairs       = [find_pairs(nodes, node_sets[a], node_sets[b]) for (a, b) in pair_sets]
  a_vectors   = first.(pairs)
  pair_groups = last.(pairs)

  # Create BEqs and B matrices
  BEqs  = makeBEqs(vcat(pair_groups...), nNodes)
  B0    = dropzeros!(makeB0(BEqs, nDoFs=nDoFs))
  B0d   = dropzeros!(makeB0(BEqs, nDoFs=1))

  # Transform a vectors
  a_vectors = [M * a for a in a_vectors]

  # Create Bϵ matrix
  Beps = if nDoFs == 2
    #     e₁₁, e₂₂, e₁₂
    a -> [a[1] 0 a[2];
          0 a[2] a[1]]
  else
    #     e₁₁, e₂₂, e₃₃, e₂₃, e₁₃, e₁₂
    a -> [a[1] 0 0 0 a[3] a[2];
          0 a[2] 0 a[3] 0 a[1];
          0 0 a[3] a[2] a[1] 0]
  end

  Ba  = makeBa(Tuple(pair_groups), nNodes)
  Bϵi = [Beps(a) for a in a_vectors]
  Bϵ  = dropzeros!(sparse(Ba * vcat(Bϵi...)))

  return (B0, B0d, Bϵ, Tuple(a_vectors))
end

function find_pairs(nodes, set1, set2; bchk = false, dTol=1e-12)
  N1,N2 = length(set1),length(set2)
  @assert N1==N2 @sprintf("length(set1)=%i!=%i=length(set2)",N1,N2)

  a = let
    cg1  = sum(nodes[set1])/N1
    cg2  = sum(nodes[set2])/N2
    cg2-cg1
  end

  pairs = Vector{Pair{Int64,Int64}}(undef, N1)
  for ii1 in 1:N1
    node1 = nodes[set1[ii1]]
    dd    = [norm(a + node1 - nodes[set2[jj]]) for jj in 1:N2]
    ii2   = argmin(dd)
    bchk && @assert dd[ii2] ≤ dTol @sprintf("mininum distance out of tolerance: %.3f ≰ %.3f",
                                            dd[ii2], dTol)
    pairs[ii1] = set1[ii1]=>set2[ii2]
  end

  a, pairs
end

function makeBEqs(all_pairs, nNodes, T=Float64)
  A = begin
    rmrow(A, ii) = A[1:size(A)[1] .!= ii, :]
    A = spzeros(T, size(all_pairs)[1], nNodes)
    for (ii, pair) = enumerate(all_pairs)
      A[ii, pair[1]] = A[ii, pair[2]] = 1
    end
    for ii=1:nNodes
      id_rows = sort(findall(A[:,ii].!=0))
      nrows   = length(id_rows)
      if nrows>1
        for jj=nrows:-1:2
          A[id_rows[1],:] = ((A[id_rows[1],:].==1) .| (A[id_rows[jj],:].==1))
          A = rmrow(A, id_rows[jj])
        end
      end
    end  
    A
  end
  B = begin
    nNodes = size(A)[2]
    q      = [sum(A[:,ii]) for ii=1:nNodes] 
    ifree  = findall(q .==0)
    nfree  = length(ifree)
    B      = spzeros(T, nfree, nNodes)
    for (ii,idx) in enumerate(ifree)
      B[ii, idx] = 1
    end  
    B
  end
  return vcat(A,B)
end
function makeB0(ufree::BitArray{N} where N, T)
  #   nDoFstot = length(ufree)
  N    = sum(ufree)
  idxx = findall(ufree[:])
  sparse(idxx, 1:N, ones(N), length(ufree), N)
end
function makeB0(BEqs::SparseMatrixCSC{TF,TI}; nDoFs=3) where {TF,TI}
  nEqs,nNodes = size(BEqs)
  nDoFstot    = nNodes*nDoFs 
  I           = zeros(TI, nDoFstot)
  J           = zeros(TI, nDoFstot)
  V           = zeros(TF, nDoFstot)

  for ii=1:nNodes
    iirows = (ii-1)*nDoFs
    for qq = BEqs[:,ii].nzind
      iicols = (qq-1)*nDoFs
      for jj = 1:nDoFs
        I[iirows+jj] = iirows+jj
        J[iirows+jj] = iicols+jj
        V[iirows+jj] = one(TF)
      end
    end
  end  
  sparse(I,J,V)
end
function makeBa(pairs, nNodes,
                T=Float64;
                nDoFsu=length(pairs),
                nDoFsω = 0)

  nDoFs = nDoFsu + nDoFsω
  ndirs = length(pairs)
  Ba    = spzeros(T, nNodes*nDoFs, nDoFsu*ndirs)

  for (jj, pairs) in enumerate(pairs)
    for pair in pairs
      iia = (jj-1)*nDoFsu
      ii1 = (pair[1]-1)*nDoFs
      ii2 = (pair[2]-1)*nDoFs
      for ii=1:nDoFsu
        Ba[ii1+ii, iia+ii] = -1/2
        Ba[ii2+ii, iia+ii] = 1/2
      end
    end
  end
  dropzeros!(Ba)
end

function getT(elem::C2D{P,<:Any}, udot0::Matrix{T}) where {T,P}
  ϕ   = zero(T) 
  for ii=1:P
    N = elem.N[ii]
    d  = [N⋅udot0[1:2:end], N⋅udot0[2:2:end]]
    ϕ += elem.mat.mat.ρ*elem.wgt[ii]* (d⋅d)
  end
  ϕ
end
function getT(elems::Vector{<:CElem}, udot::Matrix{T}) where T
  nElems = length(elems)
  Φ = Vector{adiff.D2}(undef, nElems)
  Threads.@threads for ii=1:nElems
    Φ[ii] = getT(elems[ii], adiff.D2(udot[:,elems[ii].nodes]))
  end

  makeϕrKt(Φ, elems, udot)
end



end
