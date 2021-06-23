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
        for (n, I) in zip(LinearIndices(A), CartesianIndices(A))
            @test A[n] === A[I]
        end

        for (n, v) in pairs(IndexLinear(), A)
            @test A[n] === v
        end

        for (I, v) in pairs(IndexCartesian(), A)
            @test A[I] === v
        end
    end
end
