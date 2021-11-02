using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
myid = rank + 1

pen = Pencil((16, 32, 14), comm)
u = PencilArray{Int32}(undef, pen)
fill!(u, 2myid)

@testset "Reductions" begin
    @test minimum(u) == 2
    @test maximum(u) == 2nprocs
    @test minimum(abs2, u) == 2^2
    @test maximum(abs2, u) == (2nprocs)^2
    @test sum(u) === MPI.Allreduce(sum(parent(u)), +, comm)
    @test sum(abs2, u) === MPI.Allreduce(sum(abs2, parent(u)), +, comm)

    # These exact equalities work because we're using integers.
    # They are not guaranteed to work with floats.
    @test foldl(min, u) === minimum(u)
    @test mapfoldl(abs2, min, u) === minimum(abs2, u)

    @test foldr(min, u) === minimum(u)
    @test mapfoldr(abs2, min, u) === minimum(abs2, u)
end
