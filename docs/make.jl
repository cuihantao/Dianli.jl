using Dianli
using Documenter

makedocs(;
    modules=[Dianli],
    authors="Hantao Cui <cuihantao@gmail.com> and contributors",
    repo="https://github.com/cuihantao/Dianli.jl/blob/{commit}{path}#L{line}",
    sitename="Dianli.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cuihantao.github.io/Dianli.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cuihantao/Dianli.jl",
)
