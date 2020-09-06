# This is based on the runtests.jl file of MPI.jl.

using MPI: mpiexec

test_files = [
    "broadcast.jl",
    "pencils.jl",
    "hdf5.jl",
]

Nproc = clamp(Sys.CPU_THREADS, 4, 8)

for fname in test_files
    @info "Running $fname with $Nproc processes..."
    mpiexec() do cmd
        # Disable precompilation to prevent race conditions when loading
        # packages.
        run(`$cmd -n $Nproc $(Base.julia_cmd()) --compiled-modules=no $fname`)
    end
    println()
end
