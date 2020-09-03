using MPI
using HDF5  # to load HDF5 code via Requires
using PencilArrays
using Documenter

DocMeta.setdocmeta!(PencilArrays.Permutations, :DocTestSetup,
                    :(using PencilArrays.Permutations); recursive=true)

function main()
    fastmode = get(ENV, "JULIA_DOCS_FAST", "") ∈ ("1", "true", "yes")
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    keepdocs = rank == 0  # the docs from this process will be kept and deployed

    mktempdir() do tempdir
        makedocs(;
            modules = [PencilArrays],
            authors = "Juan Ignacio Polanco <jipolanc@gmail.com> and contributors",
            repo = "https://github.com/jipolanco/PencilArrays.jl/blob/{commit}{path}#L{line}",
            sitename = "PencilArrays.jl",
            build = keepdocs ? "build" : tempdir,
            format = Documenter.HTML(;
                prettyurls = true,
                canonical = "https://jipolanco.github.io/PencilArrays.jl",
                assets = [
                    "assets/custom.css",
                    "assets/matomo.js",
                ],
            ),
            pages = [
                "Home" => "index.md",
                "Library" => [
                    "MPITopology.md",
                    "Pencils.md",
                    "PencilArrays.md",
                    "Transpositions.md",
                    "PencilIO.md",
                    "PencilArrays_timers.md",
                    "Internals" => ["PermutationUtils.md"]
                ],
            ],
            linkcheck = keepdocs && !fastmode,
        )
    end

    if keepdocs
        deploydocs(;
            repo = "github.com/jipolanco/PencilArrays.jl",
            forcepush = true,
        )
    end

    nothing
end

MPI.Initialized() || MPI.Init()

main()
