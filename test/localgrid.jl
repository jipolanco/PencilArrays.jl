using MPI
using PencilArrays
using Test

using PencilArrays.LocalGrids:
    LocalRectilinearGrid, components

# TODO
# - return SVector for grid elements?

MPI.Init()
comm = MPI.COMM_WORLD
MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

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
    xl, yl, zl = @inferred components(grid)
    @test grid.x == xl
    @test all(i -> xl[i] == grid.x[i], eachindex(grid.x))
    @test all(i -> yl[i] == grid.y[i], eachindex(grid.y))
    @test all(i -> zl[i] == grid.z[i], eachindex(grid.z))

    # Broadcasting
    u = PencilArray{Float32}(undef, pen)
    @test @inferred(localgrid(u, coords_global)) === grid
    @test_nowarn @. u = grid.x * grid.y + grid.z

    # Indexing
    @test @inferred(eachindex(grid)) === @inferred(CartesianIndices(grid))
    @test eachindex(grid) === CartesianIndices(u)
    @test all(eachindex(grid)) do I
        x, y, z = grid[I]
        u[I] ≈ x * y + z
    end

    # Iteration: check that grids and arrays iterate in the same order
    @test all(zip(u, grid)) do (v, xyz)
        x, y, z = xyz
        v ≈ x * y + z
    end

    @test all(enumerate(grid)) do (i, xyz)
        x, y, z = xyz
        u[i] ≈ x * y + z
    end
    coords_col = @inferred collect(grid)
    @test coords_col isa Vector
    @test eltype(coords_col) === eltype(grid) ===
        typeof(map(first, components(grid)))  # = Tuple{Float64, Float64, Int}
end
