module PencilArrays

using MPI
using OffsetArrays
using Reexport
using StaticPermutations
using TimerOutputs

import Base: @propagate_inbounds
import LinearAlgebra

include("Pencils/Pencils.jl")
@reexport using .Pencils

include("PermutedIndices/PermutedIndices.jl")
using .PermutedIndices

include("LocalGrids/LocalGrids.jl")
@reexport using .LocalGrids

import .Pencils:
    get_comm,
    permutation,
    range_local,
    range_remote,
    size_local,
    size_global,
    length_local,
    topology,
    typeof_array

export PencilArray, GlobalPencilArray, PencilArrayCollection, ManyPencilArray
export pencil
export gather
export global_view
export ndims_extra, ndims_space, extra_dims, sizeof_global

# Type definitions
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

end
