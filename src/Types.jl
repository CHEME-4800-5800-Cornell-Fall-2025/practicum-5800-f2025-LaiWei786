# Define the Classical Hopfield Network type
"""
    MyClassicalHopfieldNetworkModel

A mutable struct representing a Classical Hopfield Network.

# Fields
- `W::Matrix{Float32}`: Weight matrix of size (N, N) encoding the memories
- `b::Vector{Float32}`: Bias vector of size N (typically zeros)
- `energy::Dict{Int,Float32}`: Energy dictionary for each stored memory
"""
mutable struct MyClassicalHopfieldNetworkModel
    W::Matrix{Float32}           # Weight matrix (N × N)
    b::Vector{Float32}           # Bias vector (N)
    energy::Dict{Int,Float32}    # Energy of each stored memory
end

# Default constructor
"""
    MyClassicalHopfieldNetworkModel(N::Int)

Create an empty Hopfield network with N neurons.
"""
function MyClassicalHopfieldNetworkModel(N::Int)
    W = zeros(Float32, N, N)
    b = zeros(Float32, N)
    energy = Dict{Int,Float32}()
    return MyClassicalHopfieldNetworkModel(W, b, energy)
end