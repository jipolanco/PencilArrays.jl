using Random

function Random.rand!(rng::AbstractRNG, u::PencilArray)
    rand!(rng, parent(u))
    u
end

# This is to workaround scalar indexing issue with GPUArrays (or at least with JLArrays).
# GPUArrays.jl defines rand!(::AbstractRNG, ::AnyGPUArray) but not rand!(::AnyGPUArray),
# which ends up calling a generic rand! implementation in Julia base.
Random.rand!(u::PencilArray) = rand!(Random.default_rng(), u)

function Random.randn!(rng::AbstractRNG, u::PencilArray, args...)
    randn!(rng, parent(u), args...)
    u
end
