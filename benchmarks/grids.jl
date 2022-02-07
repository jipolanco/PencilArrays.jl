using MPI
using PencilArrays
using BenchmarkTools
using Random

using PencilArrays.LocalGrids:
    LocalRectilinearGrid, components

MPI.Init()
comm = MPI.COMM_WORLD
MPI.Comm_rank(comm) == 0 || redirect_stdout(devnull)

@inline ftest(u, x, y, z) = u * (x + 2y + z^2)

function bench_eachindex!(v, u, grid)
    for I ∈ eachindex(grid)
        @inbounds v[I] = ftest(u[I], grid[I]...)
    end
    v
end

function bench_iterators!(v, u, grid)
    for (n, xyz) ∈ zip(eachindex(u), grid)
        @inbounds v[n] = ftest(u[n], xyz...)
    end
    v
end

function bench_rawcoords!(v, u, coords)
    for (n, I) ∈ zip(eachindex(u), CartesianIndices(u))
        @inbounds xyz = map(getindex, coords, Tuple(I))
        @inbounds v[n] = ftest(u[n], xyz...)
    end
    v
end

function benchmark_pencil(pen)
    println(pen, "\n")
    dims = size(pen)

    # Note: things are roughly twice as fast if one "collects" ranges into
    # regular arrays.
    coords_global = map(
        xs -> collect(Float64, xs),
        (
            range(0, 1; length = dims[1]),
            range(0, 1; length = dims[2]),
            [n^2 for n = 1:dims[3]],
        )
    )

    grid = localgrid(pen, coords_global)

    coords_local = map(view, coords_global, range_local(pen, LogicalOrder()))
    @assert components(grid) == coords_local

    u = PencilArray{Float64}(undef, pen)
    randn!(u)

    v = similar(u)

    print("- Broadcast: ")
    @btime $v .= ftest.($u, $(grid.x), $(grid.y), $(grid.z))

    vcopy = copy(v)

    fill!(v, 0)
    print("- Eachindex: ")
    @btime bench_eachindex!($v, $u, $grid)
    @assert v == vcopy

    fill!(v, 0)
    print("- Iterators: ")
    @btime bench_iterators!($v, $u, $grid)
    @assert v == vcopy

    fill!(v, 0)
    print("- Raw coords:")  # i.e. without localgrid
    @btime bench_rawcoords!($v, $u, $coords_local)
    @assert v == vcopy

    nothing
end

dims = (60, 110, 21)
perms = [NoPermutation(), Permutation(2, 3, 1)]

for (n, perm) ∈ enumerate(perms)
    s = perm == NoPermutation() ? "Without permutations" : "With permutations"
    println("\n($n) ", s, "\n")
    pen = Pencil(dims, comm; permute = perm)
    benchmark_pencil(pen)
end

#=============================================================

Benchmark results
=================

On Julia 1.7.2 + PencilArrays v0.15.0 and 1 MPI process.

This is with --check-bounds=no.
Without that flag, things are a bit slower for the "Iterators" and "Raw coords"
cases, which probably means that there are some @inbounds missing somewhere in
the code.

(1) Without permutations

Decomposition of 3D data
    Data dimensions: (60, 110, 21)
    Decomposed dimensions: (2, 3)
    Data permutation: NoPermutation()
    Array type: Array

- Broadcast:   212.889 μs (0 allocations: 0 bytes)
- Eachindex:   171.430 μs (0 allocations: 0 bytes)
- Iterators:   182.775 μs (0 allocations: 0 bytes)
- Raw coords:  205.575 μs (0 allocations: 0 bytes)

(2) With permutations

Decomposition of 3D data
    Data dimensions: (60, 110, 21)
    Decomposed dimensions: (2, 3)
    Data permutation: Permutation(2, 3, 1)
    Array type: Array

- Broadcast:   216.302 μs (0 allocations: 0 bytes)
- Eachindex:   175.397 μs (0 allocations: 0 bytes)
- Iterators:   158.978 μs (0 allocations: 0 bytes)
- Raw coords:  312.931 μs (0 allocations: 0 bytes)

=============================================================#
