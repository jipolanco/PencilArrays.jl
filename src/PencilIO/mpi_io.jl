export MPIIODriver

"""
    MPIIODriver(; sequential=false, uniqueopen=false, deleteonclose=false)

MPI-IO driver using the MPI.jl package.

Keyword arguments are passed to
[`MPI.File.open`](https://juliaparallel.github.io/MPI.jl/latest/io/#MPI.File.open).
"""
struct MPIIODriver <: ParallelIODriver
    sequential :: Bool
    uniqueopen :: Bool
    deleteonclose :: Bool
    MPIIODriver(; sequential=false, uniqueopen=false, deleteonclose=false) =
        new(sequential, uniqueopen, deleteonclose)
end

Base.open(D::MPIIODriver, filename::AbstractString, comm::MPI.Comm; keywords...) =
    MPI.File.open(comm, filename;
                  sequential=D.sequential, uniqueopen=D.uniqueopen,
                  deleteonclose=D.deleteonclose, keywords...)
