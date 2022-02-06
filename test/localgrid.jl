using MPI
using PencilArrays
using Test

using PencilArrays.LocalGrids:
    LocalRectilinearGrid, components

MPI.Init()
comm = MPI.COMM_WORLD

perm = Permutation(2, 3, 1)
@assert inv(perm) != perm

dims = (6, 11, 21)
pen = Pencil(dims, comm; permute = perm)

@testset "LocalRectilinearGrid" begin
    coords_global = (
        range(0, 1; length = dims[1]),
        range(0, 1; length = dims[2]),
        [n^2 for n = 1:dims[3]],  # these are Int
    )

    grid = @inferred localgrid(pen, coords_global)
    @test grid isa LocalRectilinearGrid{3}
    @test permutation(grid) === permutation(pen)
    @test ndims(grid) == 3
    @test match(
        r"^LocalRectilinearGrid\{3\} with Permutation\(.*\) and coordinates:",
        repr(grid),
    ) !== nothing

    # Components
    @inferred (g -> (g.x, g.y, g.z))(grid)
    @inferred (g -> (g[1], g[2], g[3]))(grid)
    @test match(
        r"^Component i = 2 of LocalRectilinearGrid\{3\}:", repr(grid.y),
    ) !== nothing

    # Broadcasting
    u = PencilArray{Float32}(undef, pen)
    @test_nowarn @. u = grid.x * grid.y + grid.z

    # Iteration
    coords_col = @inferred collect(grid)
    @test coords_col isa Vector
    @test eltype(coords_col) === eltype(grid) ===
        typeof(map(first, components(grid)))  # = Tuple{Float64, Float64, Int}

    # Indexing
end
