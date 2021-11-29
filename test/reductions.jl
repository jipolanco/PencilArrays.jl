using MPI
using PencilArrays
using Test

MPI.Initialized() || MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
myid = rank + 1

# This can be simplified to `redirect_stdout(devnull)` on Julia ≥ 1.6.
let dev_null = @static Sys.iswindows() ? "nul" : "/dev/null"
    MPI.Comm_rank(comm) == 0 || redirect_stdout(open(dev_null, "w"))
end

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
