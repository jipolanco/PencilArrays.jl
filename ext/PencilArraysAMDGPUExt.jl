module PencilArraysAMDGPUExt

using PencilArrays: typeof_array, typeof_ptr
using PencilArrays.Transpositions: Transpositions
using AMDGPU: ROCVector

# Workaround `unsafe_wrap` not allowing the `own` keyword argument in the AMDGPU
# implementation.
# Moreover, one needs to set the `lock = false` argument to indicate that we want to wrap an
# array which is already in the GPU.
function Transpositions.unsafe_as_array(::Type{T}, x::ROCVector{UInt8}, dims::Tuple) where {T}
    p = typeof_ptr(x){T}(pointer(x))
    unsafe_wrap(typeof_array(x), p, dims; lock = false)
end

# Workaround `unsafe_wrap` for ROCArrays not providing a definition for dims::Integer.
# We convert that argument to a tuple, which is accepted by the implementation in AMDGPU.
function Transpositions.unsafe_as_array(::Type{T}, x::ROCVector{UInt8}, N::Integer) where {T}
    Transpositions.unsafe_as_array(T, x, (N,))
end

end
