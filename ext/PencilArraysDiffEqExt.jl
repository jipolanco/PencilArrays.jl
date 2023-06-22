module PencilArraysDiffEqExt

using PencilArrays: PencilArray, length_global
using DiffEqBase

# This is used for adaptive timestepping in DifferentialEquations.jl.
# Without this, each MPI process may choose a different dt, leading to
# catastrophic consequences!
DiffEqBase.recursive_length(u::PencilArray) = length_global(u)

end
