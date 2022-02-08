using HDF5  # to load HDF5 code via Requires
using PencilArrays
using Documenter

DocMeta.setdocmeta!(
    PencilArrays, :DocTestSetup,
    quote
        using PencilArrays
        using MPI
        MPI.Initialized() || MPI.Init()
    end;
    recursive=true,
)

doctest(PencilArrays; fix = false)

function main()
    makedocs(;
        modules = [PencilArrays],
        authors = "Juan Ignacio Polanco <jipolanc@gmail.com> and contributors",
        repo = "https://github.com/jipolanco/PencilArrays.jl/blob/{commit}{path}#L{line}",
        sitename = "PencilArrays.jl",
        format = Documenter.HTML(;
            prettyurls = true,
            canonical = "https://jipolanco.github.io/PencilArrays.jl",
            assets = [
                "assets/custom.css",
                "assets/tomate.js",
            ],
        ),
        pages = [
            "Home" => "index.md",
            "Library" => [
                "Pencils.md",
                "PencilArrays.md",
                "LocalGrids.md",
                "Transpositions.md",
                "PencilIO.md",
                "MPITopology.md",
                "PencilArrays_timers.md",
            ],
            "Additional notes" => [
                "notes/reductions.md",
            ],
        ],
        linkcheck = false,
    )

    deploydocs(;
        repo = "github.com/jipolanco/PencilArrays.jl",
        forcepush = true,
    )

    nothing
end

main()
