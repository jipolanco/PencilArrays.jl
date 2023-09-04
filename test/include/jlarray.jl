import JLArrays

# Define a few more functions needed for PencilArrays tests
# (these seem to be defined for CuArray in CUDA.jl)
# TODO define these in JLArrays.jl

using Random: Random, AbstractRNG
using JLArrays: DenseJLVector, JLArray

Base.resize!(u::DenseJLVector, n) = (resize!(u.data, n); u)

function Base.unsafe_wrap(::Type{JLArray}, p::Ptr, dims::Union{Integer, Dims}; kws...)
    data = unsafe_wrap(Array, p, dims; kws...)
    JLArray(data)
end

function Random.rand!(rng::AbstractRNG, u::JLArray, ::Type{X}) where {X}
    rand!(rng, u.data, X)
    u
end
