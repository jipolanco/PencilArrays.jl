using Random

function Random.rand!(rng::AbstractRNG, u::PencilArray, sp::Random.Sampler)
    rand!(rng, parent(u), sp)
    u
end

function Random.randn!(rng::AbstractRNG, u::PencilArray, args...)
    randn!(rng, parent(u), args...)
    u
end
