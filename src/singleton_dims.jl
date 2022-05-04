# Internal type specifying a list of singleton dimensions, e.g. singleton = (1, 3).
struct SingletonDims{D <: Dims}
    dims :: D
    function SingletonDims(d::Tuple)
        dims = convert(Dims, d)
        new{typeof(dims)}(dims)
    end
end

SingletonDims() = SingletonDims(())
SingletonDims(s::SingletonDims) = s
SingletonDims(i::Integer) = SingletonDims((i,))

# Replaces singleton dimensions by 1.
# Example: singletons_to_one((3, 4, 12), (1, 3)) -> (1, 4, 1)
@inline function singletons_to_one(dims, singleton)
    for i ∈ singleton
        dims = Base.setindex(dims, 1, i)
    end
    dims
end

@inline singletons_to_one(dims, ::NTuple{0}) = dims
@inline singletons_to_one(dims, s::SingletonDims) = singletons_to_one(dims, s.dims)
