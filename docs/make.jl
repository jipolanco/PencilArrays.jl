using PencilArrays
using Documenter
using Documenter.Remotes: GitHub

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
        authors = "Juan Ignacio Polanco <juan-ignacio.polanco@cnrs.fr>",
        repo = GitHub("jipolanco", "PencilArrays.jl"),
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
        warnonly = [:missing_docs],  # TODO can we remove this?
    )

    deploydocs(;
        repo = "github.com/jipolanco/PencilArrays.jl",
        forcepush = true,
        # PRs deploy at https://jipolanco.github.io/PencilArrays.jl/previews/PR**
        push_preview = true,
    )

    nothing
end

main()
