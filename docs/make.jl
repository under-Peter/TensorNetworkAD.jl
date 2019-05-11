using Documenter, TensorNetworkAD

makedocs(;
    modules=[TensorNetworkAD],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/under-Peter/TensorNetworkAD.jl/blob/{commit}{path}#L{line}",
    sitename="TensorNetworkAD.jl",
    authors="Andreas Peter",
    assets=[],
)

deploydocs(;
    repo="github.com/under-Peter/TensorNetworkAD.jl",
)
