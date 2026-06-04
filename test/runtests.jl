# Avoid precompilation in parallel
using Pkg
Pkg.precompile()

using MPIPreferences
@show MPIPreferences.binary

using InteractiveUtils
using MPI: MPI, mpiexec
using PencilArrays

# These tests can be run in serial mode
test_files_serial = [
    "permutations.jl",
]

test_files = [
    "localgrid.jl",
    "io.jl",
    "reductions.jl",
    "broadcast.jl",
    "array_types.jl",
    "arrays.jl",
    "pencils.jl",
    "transpose.jl",
    "array_interface.jl",
    "adapt.jl",
    "ode.jl",
]

Nproc = let N = get(ENV, "JULIA_MPI_TEST_NPROCS", nothing)
    N === nothing ? clamp(Sys.CPU_THREADS, 4, 6) : parse(Int, N)
end

println()
versioninfo()
MPI.versioninfo()

if MPIPreferences.binary != "system"
    error("""
        tests should be run with system MPI binaries for testing parallel HDF5
        (found MPIPreferences.binary = $(MPIPreferences.binary))
        """)
end

for fname in test_files_serial
    include(fname)
end

for fname in test_files
    @info "Running $fname with $Nproc processes..." Base.julia_cmd()
    run(`$(mpiexec()) -n $Nproc $(Base.julia_cmd()) $fname`)
    println()
end
