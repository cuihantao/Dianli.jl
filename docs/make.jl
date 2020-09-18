using Dian
using Documenter

makedocs(;
    modules=[Dian],
    authors="Hantao Cui <cuihantao@gmail.com> and contributors",
    repo="https://github.com/cuihantao/Dian.jl/blob/{commit}{path}#L{line}",
    sitename="Dian.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cuihantao.github.io/Dian.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cuihantao/Dian.jl",
)
