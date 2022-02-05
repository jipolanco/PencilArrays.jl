using MPI
using PencilArrays
using Random
using Test

MPI.Initialized() || MPI.Init()

dims = (8, 12, 9)
comm = MPI.COMM_WORLD
pen = Pencil(dims, comm; permute = Permutation(2, 3, 1))

MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

u = PencilArray{Float32}(undef, pen)
randn!(u)
ug = global_view(u)

@testset "Indices" begin
    for A in (u, ug)
        @test all(zip(LinearIndices(A), CartesianIndices(A))) do (n, I)
            A[n] === A[I]
        end

        @test all(pairs(IndexLinear(), A)) do (n, v)
            A[n] === v
        end

        @test all(pairs(IndexCartesian(), A)) do (I, v)
            A[I] === v
        end
    end
end
