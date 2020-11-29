# This is based on the runtests.jl file of MPI.jl.

using MPI: mpiexec

test_files = [
    "array_interface.jl",
    "broadcast.jl",
    "io.jl",
    "pencils.jl",
]

Nproc = let N = get(ENV, "JULIA_MPI_NPROC", nothing)
    N === nothing ? clamp(Sys.CPU_THREADS, 4, 8) : parse(Int, N)
end

for fname in test_files
    @info "Running $fname with $Nproc processes..."
    mpiexec() do cmd
        # Disable precompilation to prevent race conditions when loading
        # packages.
        run(`$cmd -n $Nproc $(Base.julia_cmd()) --compiled-modules=no $fname`)
    end
    println()
end
