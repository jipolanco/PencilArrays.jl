# Work around https://github.com/JuliaLang/Pkg.jl/issues/2500 for CI on Julia ≤ 1.7.
# (Adapted from https://github.com/JuliaParallel/MPI.jl/pull/564, which was not merged.)
if VERSION ≤ v"1.8-"
    test_project = first(Base.load_path())
    # @__DIR__ is something like "~/.julia/dev/PencilArrays/test"
    # We look for LocalPreferences.toml in the PencilArrays directory.
    preferences_file = joinpath(@__DIR__, "..", "LocalPreferences.toml")
    # This is something like "/tmp/jl_NcTbc4/LocalPreferences.toml"
    test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
    if isfile(preferences_file) && !isfile(test_preferences_file)
        cp(preferences_file, test_preferences_file)
    end
end

using MPIPreferences
@show MPIPreferences.binary

using InteractiveUtils
using MPI: MPI, mpiexec

# Load test packages to trigger precompilation
using PencilArrays

# These tests can be run in serial mode
test_files_serial = [
    "permutations.jl",
]

test_files = [
    "io.jl",
    "localgrid.jl",
    "reductions.jl",
    "broadcast.jl",
    "array_types.jl",
    "arrays.jl",
    "pencils.jl",
    "array_interface.jl",
    "adapt.jl",
    "ode.jl",
]

Nproc = let N = get(ENV, "JULIA_MPI_TEST_NPROCS", nothing)
    N === nothing ? clamp(Sys.CPU_THREADS, 4, 6) : parse(Int, N)
end

println()
versioninfo()
if isdefined(MPI, :versioninfo)
    MPI.versioninfo()
else
    println("\n", MPI.MPI_LIBRARY_VERSION_STRING, "\n")
end

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
    @info "Running $fname with $Nproc processes..."
    mpiexec() do cmd
        run(`$cmd -n $Nproc $(Base.julia_cmd()) $fname`)
    end
    println()
end
