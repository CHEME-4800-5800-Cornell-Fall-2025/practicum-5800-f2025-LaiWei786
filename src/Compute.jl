# Compute methods for Hopfield network operations

"""
    hamming(state1::Vector, state2::Vector)

Compute the Hamming distance between two binary vectors.

H(a, b) = Σ_i I[a_i ≠ b_i]

# Arguments
- `state1::Vector`: First binary vector
- `state2::Vector`: Second binary vector

# Returns
- `Int`: Number of differing positions
"""
function hamming(state1::Vector, state2::Vector)
    return sum(state1 .!= state2)
end

"""
    decode(state::Vector)

Decode a binary state vector back to an image matrix.

# Arguments
- `state::Vector{Int32}`: Flattened binary state vector (±1 values)

# Returns
- `Matrix{Float32}`: 28×28 image matrix with values in [0, 1]
"""
function decode(state::Vector{Int32})
    N = length(state)
    # Assume 28×28 images
    rows = 28
    cols = 28
    
    img = zeros(Float32, rows, cols)
    linear_idx = 1
    
    for row in 1:rows
        for col in 1:cols
            if linear_idx <= N
                # Convert from ±1 to 0/1 range
                img[row, col] = (state[linear_idx] + 1) / 2.0f0
            end
            linear_idx += 1
        end
    end
    
    return img
end

"""
    recover(model::MyClassicalHopfieldNetworkModel, initial_state::Vector, 
            true_energy::Float32; maxiterations::Int=1000, patience::Union{Int,Nothing}=nothing,
            miniterations_before_convergence::Union{Int,Nothing}=nothing)

Recover a pattern from the network using asynchronous updates.

# Arguments
- `model::MyClassicalHopfieldNetworkModel`: The Hopfield network
- `initial_state::Vector`: Initial (noisy) state (±1 values)
- `true_energy::Float32`: Energy of the target memory (for reference)
- `maxiterations::Int`: Maximum number of iterations (default: 1000)
- `patience::Union{Int,Nothing}`: Number of identical consecutive states for convergence (default: 5)
- `miniterations_before_convergence::Union{Int,Nothing}`: Minimum iterations before checking convergence

# Returns
- `frames::Dict{Int64, Vector{Int32}}`: Dictionary mapping iteration → network state
- `energydictionary::Dict{Int64, Float32}`: Dictionary mapping iteration → network energy
"""
function recover(model::MyClassicalHopfieldNetworkModel, initial_state::Vector, 
                true_energy::Float32; maxiterations::Int=1000, patience::Union{Int,Nothing}=nothing,
                miniterations_before_convergence::Union{Int,Nothing}=nothing)
    
    N = length(initial_state)
    
    if isnothing(patience)
        patience = 5
    end
    
    if isnothing(miniterations_before_convergence)
        miniterations_before_convergence = patience
    end
    
    # Initialize state (convert to Float32)
    s = convert(Vector{Float32}, initial_state)
    
    # Initialize tracking structures
    frames = Dict{Int64, Vector{Int32}}()
    energydictionary = Dict{Int64, Float32}()
    state_history = Vector{Vector{Float32}}()
    
    converged = false
    t = 1
    
    while !converged && t <= maxiterations
        # Asynchronous update: choose a random neuron
        i = rand(1:N)
        
        # Compute new state for neuron i
        activation = sum(model.W[i, j] * s[j] for j in 1:N) - model.b[i]
        s_new = sign(activation)
        if s_new == 0  # Handle sign(0) = 0 case
            s_new = 1.0f0
        end
        s[i] = s_new
        
        # Store state and energy
        frames[t] = convert(Vector{Int32}, sign.(s))
        energydictionary[t] = -0.5f0 * dot(s, model.W * s) - dot(model.b, s)
        
        # Store in history for convergence checking
        push!(state_history, copy(s))
        if length(state_history) > patience
            popfirst!(state_history)
        end
        
        # Check for convergence
        if t >= miniterations_before_convergence && length(state_history) >= patience
            # Check if all states in history are identical
            all_same = true
            for j in 2:patience
                if hamming(state_history[j], state_history[1]) != 0
                    all_same = false
                    break
                end
            end
            
            if all_same
                converged = true
            end
        end
        
        t += 1
    end
    
    return frames, energydictionary
end
