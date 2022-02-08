```@meta
CurrentModule = PencilArrays.LocalGrids
```

# Working with grids

PencilArrays.jl includes functionality for conveniently working with the
coordinates associated to a multidimensional domain.
For this, the [`localgrid`](@ref) function can be used to construct an object
describing the grid coordinates associated to the local MPI process.
This object can be used to easily and efficiently perform operations that
depend on the local coordinates.

## Creating local grids

As an example, say we are performing a 3D simulation on a [rectilinear
grid](https://en.wikipedia.org/wiki/Regular_grid#Rectilinear_grid), so that
the coordinates of a grid point are given by ``\bm{x}_{ijk} = (x_i, y_j, z_k)``
where `x`, `y` and `z` are separate one-dimensional arrays.
For instance:

```@example LocalGrids
Nx, Ny, Nz = 65, 17, 21
xs = range(0, 1; length = Nx)
ys = range(-1, 1; length = Ny)
zs = range(0, 2; length = Nz)
nothing # hide
```

Before continuing, let's create a domain decomposition configuration:

```@example LocalGrids
using MPI
using PencilArrays

MPI.Init()
comm = MPI.COMM_WORLD

pen = Pencil((Nx, Ny, Nz), comm)
```

Now, we can extract the local grid associated to the local MPI process:

```@example LocalGrids
grid = localgrid(pen, (xs, ys, zs))
```

Note that this example was run on a single MPI process, which makes things
somewhat less interesting, but the same applies to more processes.
With more than one process, the local grid is a subset of the global grid
defined by the coordinates `(xs, ys, zs)`.

## Using local grids

The `grid` object just created can be used to operate with `PencilArray`s.
In particular, say we want to initialise a `PencilArray` to a function that
depends on the domain coordinates, ``u(x, y, z) = x + 2y + z^2``.
This can be easily done using the broadcasting syntax (here we use the `@.`
macro for convenience):

```@example LocalGrids
u = PencilArray{Float64}(undef, pen)  # construct PencilArray first
@. u = grid.x + 2 * grid.y + grid.z^2
nothing # hide
```

Here, `grid.x`, `grid.y` and `grid.z` are a convenient way of extracting the
three components of the grid.
Alternatively, one can use the syntax `grid[1]`, `grid[2]`, etc..., which is in
particularly useful when working in dimensions higher than 3.

Note that one could do the same as above using indexing instead of
broadcasting:

```@example LocalGrids
for I ∈ eachindex(grid)
    x, y, z = grid[I]
    u[I] = x + 2y + z^2
end
```

## Library

```@docs
AbstractLocalGrid
LocalRectilinearGrid
localgrid
components
```

## Index

```@index
Pages = ["LocalGrids.md"]
```
