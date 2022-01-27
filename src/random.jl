using Random

function Random.rand!(rng::AbstractRNG, u::PencilArray, ::Type{X}) where {X}
    rand!(rng, parent(u), X)
    u
end

function Random.randn!(rng::AbstractRNG, u::PencilArray, args...)
    randn!(rng, parent(u), args...)
    u
end
