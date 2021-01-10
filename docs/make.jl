using ContextualBandits
using Documenter

makedocs(;
    modules=[ContextualBandits],
    authors="Douglas Corbin <dfcorbin98@gmail.com>",
    repo="https://github.com/dfcorbin/ContextualBandits.jl/blob/{commit}{path}#L{line}",
    sitename="ContextualBandits.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
