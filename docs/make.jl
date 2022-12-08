using SchottkyAnoMaLy
using Documenter

DocMeta.setdocmeta!(SchottkyAnoMaLy, :DocTestSetup, :(using SchottkyAnoMaLy); recursive=true)

makedocs(;
    modules=[SchottkyAnoMaLy],
    authors="W. Joe Meese <meese022@umn.edu> and contributors",
    repo="https://github.com/meese-wj/SchottkyAnoMaLy.jl/blob/{commit}{path}#{line}",
    sitename="SchottkyAnoMaLy.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://meese-wj.github.io/SchottkyAnoMaLy.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/meese-wj/SchottkyAnoMaLy.jl",
    devbranch="main",
)
