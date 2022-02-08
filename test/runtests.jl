# This is based on the runtests.jl file of MPI.jl.

using InteractiveUtils
using MPI: MPI, mpiexec

# Load test packages to trigger precompilation
using PencilArrays

# These tests can be run in serial mode
include("permutations.jl")

test_files = [
    "localgrid.jl",
    "reductions.jl",
    "broadcast.jl",
    "array_types.jl",
    "arrays.jl",
    "pencils.jl",
    "array_interface.jl",
    "io.jl",
]

Nproc = let N = get(ENV, "JULIA_MPI_TEST_NPROCS", nothing)
    N === nothing ? clamp(Sys.CPU_THREADS, 4, 6) : parse(Int, N)
end

println()
versioninfo()
println("\n", MPI.MPI_LIBRARY_VERSION_STRING, "\n")

for fname in test_files
    @info "Running $fname with $Nproc processes..."
    mpiexec() do cmd
        run(`$cmd -n $Nproc $(Base.julia_cmd()) $fname`)
    end
    println()
end
