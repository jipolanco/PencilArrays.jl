using HDF5  # to load HDF5 code via Requires
using PencilArrays
using Documenter

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
