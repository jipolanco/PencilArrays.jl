# This is based on the runtests.jl file of MPI.jl.

using InteractiveUtils
using MPI: MPI, mpiexec

test_files = [
    "io.jl",
    "pencils.jl",
    "array_interface.jl",
    "broadcast.jl",
]

Nproc = let N = get(ENV, "JULIA_MPI_NPROC", nothing)
    N === nothing ? clamp(Sys.CPU_THREADS, 4, 8) : parse(Int, N)
end

println()
versioninfo()
println("\n", MPI.MPI_LIBRARY_VERSION_STRING, "\n")

for fname in test_files
    @info "Running $fname with $Nproc processes..."
    mpiexec() do cmd
        # Disable precompilation to prevent race conditions when loading
        # packages.
        run(`$cmd -n $Nproc $(Base.julia_cmd()) --compiled-modules=no $fname`)
    end
    println()
end
