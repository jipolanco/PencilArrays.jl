"""
    AbstractIndexOrder

Abstract type determining the ordering of dimensions of an array with possibly
permuted indices.

Subtypes are [`MemoryOrder`](@ref) and [`LogicalOrder`](@ref).
"""
abstract type AbstractIndexOrder end

"""
    MemoryOrder <: AbstractIndexOrder

Singleton type specifying that array dimensions should be given in memory (or
*permuted*) order.
"""
struct MemoryOrder <: AbstractIndexOrder end

"""
    LogicalOrder <: AbstractIndexOrder

Singleton type specifying that array dimensions should be given in logical (or
*non-permuted*) order.
"""
struct LogicalOrder <: AbstractIndexOrder end

const DefaultOrder = LogicalOrder
