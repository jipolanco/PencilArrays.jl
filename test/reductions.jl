using MPI
using PencilArrays
using Test

MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
myid = rank + 1

rank == 0 || redirect_stdout(devnull)

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

    @testset "Multiple PencilArrays" begin
        û = @. u + im * u
        v̂ = copy(û)
        a = @inferred sum(abs2, û; init = zero(eltype(û)))  # `init` needed for inference when eltype(û) = Complex{Int32}...
        # These should all be equivalent:
        b = @inferred mapreduce((x, y) -> real(x * conj(y)), +, û, v̂)
        c = @inferred sum(Base.splat((x, y) -> real(x * conj(y))), zip(û, v̂))
        d = @inferred sum(xs -> real(xs[1] * conj(xs[2])), zip(û, v̂))
        @test a == b == c == d
    end

    # These exact equalities work because we're using integers.
    # They are not guaranteed to work with floats.
    @test foldl(min, u) === minimum(u)
    @test mapfoldl(abs2, min, u) === minimum(abs2, u)

    @test foldr(min, u) === minimum(u)
    @test mapfoldr(abs2, min, u) === minimum(abs2, u)
end
