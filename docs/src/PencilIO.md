# [Parallel I/O](@id PencilIO_module)

```@meta
CurrentModule = PencilArrays.PencilIO
```

The `PencilArrays.PencilIO` module contains functions for saving and loading
[`PencilArray`](@ref)s to disk using parallel I/O.
Currently, two different output formats are supported:

- raw binary files via the MPI-IO interface;
- parallel HDF5 files.

In both cases, information on dataset sizes, names and other metadata are
included along with the binary data.

The implemented approach consists in storing the data coming from different MPI
processes in a single file.
This strategy scales better in terms of number of files, and is more
convenient, than that of storing one file per process.
However, the performance is very sensitive to the configuration of the
underlying file system.
In distributed file systems such as
[Lustre](https://en.wikipedia.org/wiki/Lustre_(file_system)), it is worth
tuning parameters such as the stripe count and stripe size.
For more information, see for instance the [Parallel HDF5
page](https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5).

## Getting started

The first step before writing `PencilArray`s is to choose the parallel I/O
driver, which determines the format of the output data.
Two different drivers are currently available:

- [`MPIIODriver`](@ref): parallel I/O via the MPI-IO API and the [MPI.jl
  wrappers](https://juliaparallel.github.io/MPI.jl/latest/io/). This driver
  writes a raw binary file, along with a JSON file describing dataset metadata
  (name, dimensions, location in file, ...);

- [`PHDF5Driver`](@ref): parallel I/O via the Parallel HDF5 API and
  [HDF5.jl](https://github.com/JuliaIO/HDF5.jl). This driver requires a special
  set-up, as detailed in the [dedicated section](@ref setting_up_parallel_hdf5).

### Writing data

To open a parallel file, pass the MPI communicator and an instance of the
chosen driver to [`open`](@ref).
For instance, the following opens an MPI-IO file in write mode:

```julia
ff = open(MPIIODriver(), "filename.bin", MPI.COMM_WORLD; write=true)
```

Datasets, in the form of `PencilArray`s, can then be written as follows:

```julia
v = PencilArray(...)
ff["velocity"] = v
```

This writing step may be customised via keyword arguments such as `chunks` and
`collective`. These options are supported by both MPI-IO and HDF5 drivers.
For instance:

```julia
ff["velocity", chunks=true, collective=false] = v
```

See [`setindex!`](@ref) for the meaning of these options for each driver, as
well as for driver-specific options.

After datasets are written, the file should be closed as usual by doing
`close(ff)`. Note that the do-block syntax is also supported, as in

```julia
open(MPIIODriver(), "filename.bin", MPI.COMM_WORLD; write=true) do ff
    ff["velocity"] = v
end
```

### Reading data

Data is loaded into an existent `PencilArray` using [`read!`](@ref).
For instance:

```julia
v = PencilArray(...)
ff = open(MPIIODriver(), "filename.bin", MPI.COMM_WORLD; read=true)
    read!(ff, v, "velocity")
end
```

Note that, for the MPI-IO driver, a `filename.bin.json` file must be present
along with the `filename.bin` file containing the binary data. The JSON file is
automatically generated when writing data with this driver.

Optional keyword arguments, such as `collective`, are also supported by
[`read!`](@ref).

## [Setting-up Parallel HDF5](@id setting_up_parallel_hdf5)

If using the [Parallel HDF5 driver](@ref PHDF5Driver), the HDF5.jl package must
be available and configured with MPI support.

Parallel HDF5 is not enabled in the default installation of HDF5.jl.
For Parallel HDF5 to work, the HDF5 C libraries wrapped by HDF5.jl must be
compiled with parallel support and linked to the specific MPI implementation
that will be used for parallel I/O.
HDF5.jl must be explicitly instructed to use parallel-enabled HDF5 libraries
available in the system.
Similarly, MPI.jl must be instructed to use the corresponding MPI libraries.
This is detailed in the sections below.

Parallel-enabled HDF5 libraries are usually included in computing clusters and
linked to the available MPI implementations.
They are also available via the package manager of a number of Linux
distributions.
(For instance, Fedora includes the `hdf5-mpich-devel` and `hdf5-openmpi-devel`
packages, respectively linked to the MPICH and OpenMPI libraries in the Fedora
repositories.)

The following step-by-step guide assumes one already has access to
parallel-enabled HDF5 libraries linked to an existent MPI installation.

### 1. Using system-provided MPI libraries

Set the environment variable `JULIA_MPI_BINARY=system` and then run
`]build MPI` from Julia.
For more control, one can also set the `JULIA_MPI_PATH` environment variable to
the top-level installation directory of the MPI library.

See the [MPI.jl
docs](https://juliaparallel.github.io/MPI.jl/stable/configuration/#Using-a-system-provided-MPI-1)
for details.

### 2. Using parallel HDF5 libraries

Set the `JULIA_HDF5_LIBRARY_PATH` environment variable to the directory where
the HDF5 libraries compiled with parallel support are found.
Then run `]build HDF5` from Julia.
Note that the selected HDF5 library must be linked to the MPI library chosen in
the previous section.
For the set-up to be persistent across HDF5.jl updates, consider setting
`JULIA_HDF5_LIBRARY_PATH` in `~/.bashrc` or similar.

See the [HDF5.jl
README](https://github.com/JuliaIO/HDF5.jl#installation) for details.

### 3. Loading PencilIO

In the `PencilIO` module, the HDF5.jl package is lazy-loaded
using [Requires](https://github.com/JuliaPackaging/Requires.jl).
This means that, in Julia code, `PencilArrays` must be loaded *after* `HDF5` for
parallel I/O functionality to be available.

The following order of `using`s ensures that parallel I/O support is available:

```julia
using MPI
using HDF5
using PencilArrays
```

## Library

```@docs
PencilIO.ParallelIODriver
MPIIODriver
PHDF5Driver
PencilIO.MPIFile
open
setindex!
read!
hdf5_has_parallel
```

## Index

```@index
Pages = ["PencilIO.md"]
```
