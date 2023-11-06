using Random

function Random.rand!(rng::AbstractRNG, u::PencilArray, args...)
    rand!(rng, parent(u), args...)
    u
end

function Random.randn!(rng::AbstractRNG, u::PencilArray, args...)
    randn!(rng, parent(u), args...)
    u
end
