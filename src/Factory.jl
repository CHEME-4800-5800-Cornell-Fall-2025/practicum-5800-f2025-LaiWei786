# Factory methods for creating and initializing Hopfield Networks

"""
    build(::Type{MyClassicalHopfieldNetworkModel}, params::NamedTuple)

Build a Hopfield network using Hebbian learning to encode multiple memories.

# Arguments
- `MyClassicalHopfieldNetworkModel`: The type to construct
- `params::NamedTuple`: Named tuple with field:
  - `memories::Matrix{Int32}`: Memory patterns stored in columns (N × K matrix)

# Returns
- `MyClassicalHopfieldNetworkModel`: A trained Hopfield network with weights computed using Hebbian learning

# Details
The weights are computed using the Hebbian learning rule:
W = (1/K) * Σ(s_i ⊗ s_i^T), where K is the number of patterns and ⊗ is the outer product
The diagonal is set to zero (no self-connections).
"""
function build(::Type{MyClassicalHopfieldNetworkModel}, params::NamedTuple)
    memories = params.memories  # N × K matrix, memories in columns
    
    N, K = size(memories)  # N = number of neurons, K = number of patterns
    
    # Initialize the model
    model = MyClassicalHopfieldNetworkModel(N)
    
    # Compute weights using Hebbian learning rule
    # W = (1/K) * Σ(s_i ⊗ s_i^T)
    W = zeros(Float32, N, N)
    for k in 1:K
        s = convert(Vector{Float32}, memories[:, k])
        W .+= s * s'  # Outer product
    end
    W ./= K  # Normalize by number of patterns
    
    # Set diagonal to zero (no self-connections)
    for i in 1:N
        W[i, i] = 0.0f0
    end
    
    model.W = W
    
    # Compute energy for each stored memory
    for k in 1:K
        s = convert(Vector{Float32}, memories[:, k])
        E = -0.5f0 * dot(s, model.W * s) - dot(model.b, s)
        model.energy[k] = E
    end
    
    return model
end
