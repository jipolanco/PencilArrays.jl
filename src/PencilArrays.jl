module PencilArrays

using MPI
using OffsetArrays
using Reexport
using StaticPermutations
using TimerOutputs
using Requires: @require

using Base: @propagate_inbounds
import Adapt

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
@reexport using .Transpositions

include("PencilIO/PencilIO.jl")

end
