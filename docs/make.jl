using SchottkyanoMaLy
using Documenter

DocMeta.setdocmeta!(SchottkyanoMaLy, :DocTestSetup, :(using SchottkyanoMaLy); recursive=true)

makedocs(;
    modules=[SchottkyanoMaLy],
    authors="W. Joe Meese <meese022@umn.edu> and contributors",
    repo="https://github.com/meese-wj/SchottkyanoMaLy.jl/blob/{commit}{path}#{line}",
    sitename="SchottkyanoMaLy.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://meese-wj.github.io/SchottkyanoMaLy.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/meese-wj/SchottkyanoMaLy.jl",
    devbranch="main",
)
