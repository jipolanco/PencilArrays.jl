using Random

function Random.rand!(rng::AbstractRNG, u::PencilArray, ::Type{S}) where {S}
    rand!(rng, parent(u), S)
    u
end

function Random.randn!(rng::AbstractRNG, u::PencilArray, args...)
    randn!(rng, parent(u), args...)
    u
end
