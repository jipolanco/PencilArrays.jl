"""
    PencilArrays

Array wrappers for MPI-distributed data.
"""
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
    get_comm, get_permutation, range_local, size_local, size_global

export PencilArray, PencilArrayCollection, ManyPencilArray
export pencil
export gather
export global_view
export ndims_extra, ndims_space, extra_dims

# Type definitions
include("arrays.jl")       # PencilArray
include("multiarrays.jl")  # ManyPencilArray
include("global_view.jl")  # GlobalPencilArray
include("cartesian_indices.jl")  # PermutedLinearIndices, PermutedCartesianIndices

include("Transpositions/Transpositions.jl")
export Transpositions

include("PencilIO/PencilIO.jl")

end