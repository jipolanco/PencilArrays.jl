using PencilArrays
using Documenter

makedocs(;
    modules=[PencilArrays],
    authors="Juan Ignacio Polanco <jipolanc@gmail.com> and contributors",
    repo="https://github.com/jipolanco/PencilArrays.jl/blob/{commit}{path}#L{line}",
    sitename="PencilArrays.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jipolanco.github.io/PencilArrays.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jipolanco/PencilArrays.jl",
)
