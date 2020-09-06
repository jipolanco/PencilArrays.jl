module PencilArrays

using MPI
using OffsetArrays
using Reexport
using TimerOutputs

import Base: @propagate_inbounds
import LinearAlgebra

include("Permutations/Permutations.jl")
using .Permutations

include("Pencils/Pencils.jl")
@reexport using .Pencils
import .Pencils:
    get_comm, get_permutation, range_local, range_remote, size_local, size_global

export PencilArray, GlobalPencilArray, PencilArrayCollection, ManyPencilArray
export pencil
export gather
export global_view
export ndims_extra, ndims_space, extra_dims

# Type definitions
include("arrays.jl")       # PencilArray
include("multiarrays.jl")  # ManyPencilArray
include("global_view.jl")  # GlobalPencilArray
include("cartesian_indices.jl")  # PermutedLinearIndices, PermutedCartesianIndices

include("broadcast.jl")

include("Transpositions/Transpositions.jl")
export Transpositions

include("PencilIO/PencilIO.jl")

end
