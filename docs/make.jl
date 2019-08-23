using Documenter, TensorNetworkAD

makedocs(;
    modules=[TensorNetworkAD],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "User Guide" => "userguide.md",
        "Doc-Strings" => "docstrings.md",
        "Future Directions" => "future.md"
    ],
    repo="https://github.com/under-Peter/TensorNetworkAD.jl/blob/{commit}{path}#L{line}",
    sitename="TensorNetworkAD.jl",
    authors="Andreas Peter",
)

deploydocs(;
    repo="github.com/under-Peter/TensorNetworkAD.jl",
)
