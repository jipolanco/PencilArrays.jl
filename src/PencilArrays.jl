module PencilArrays

using MPI
using OffsetArrays
using Reexport
using StaticPermutations
using TimerOutputs
using Requires: @require

using Base: @propagate_inbounds
import Adapt
import LinearAlgebra

include("Permutations.jl")
import .Permutations: permutation

include("PermutedIndices/PermutedIndices.jl")
using .PermutedIndices

include("LocalGrids/LocalGrids.jl")
@reexport using .LocalGrids

include("Pencils/Pencils.jl")
@reexport using .Pencils

import .Pencils:
    get_comm,
    range_local,
    range_remote,
    size_local,
    size_global,
    length_local,
    length_global,
    topology,
    typeof_array

export PencilArray, GlobalPencilArray, PencilArrayCollection, ManyPencilArray
export pencil, permutation
export gather
export global_view
export ndims_extra, ndims_space, extra_dims, sizeof_global

# Type definitions
include("singleton_dims.jl")
include("arrays.jl")       # PencilArray
include("multiarrays.jl")  # ManyPencilArray
include("global_view.jl")  # GlobalPencilArray
include("cartesian_indices.jl")  # PermutedLinearIndices, PermutedCartesianIndices
include("size.jl")

include("array_interface.jl")
include("broadcast.jl")
include("random.jl")
include("reductions.jl")
include("gather.jl")

include("Transpositions/Transpositions.jl")
export Transpositions

include("PencilIO/PencilIO.jl")

function __init__()
    @require DiffEqBase="2b5f629d-d688-5b77-993f-72d75c75574e" @eval begin
        # This is used for adaptive timestepping in DifferentialEquations.jl.
        # Without this, each MPI process may choose a different dt, leading to
        # catastrophic consequences!
        DiffEqBase.recursive_length(u::PencilArray) = length_global(u)
    end
end

end
