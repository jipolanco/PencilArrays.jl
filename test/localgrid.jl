using MPI
using PencilArrays
using Test

using PencilArrays.LocalGrids:
    LocalRectilinearGrid

MPI.Init()
comm = MPI.COMM_WORLD

perm = Permutation(2, 3, 1)
@assert inv(perm) != perm

dims = (6, 11, 21)
pen = Pencil(dims, comm; permute = perm)

coords_global = (
    range(0, 1; length = dims[1]),
    range(0, 1; length = dims[2]),
    [n^2 for n = 1:dims[3]],
)

coords_local = @inferred localgrid(pen, coords_global)
@test coords_local isa LocalRectilinearGrid{3}
@test permutation(coords_local) === permutation(pen)
@test ndims(coords_local) == 3
@test (@inferred (g -> (g.x, g.y, g.z))(coords_local)) ===
    LocalGrids.coordinates(coords_local)
@test match(
    r"^LocalRectilinearGrid{3, Permutation{.*}} with coordinates:",
    repr(coords_local),
) !== nothing
