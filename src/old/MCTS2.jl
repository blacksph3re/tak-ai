module MCTS

import ..TakInterface as Game

State = Game.Board
Action = Game.Action
HParams = Dict{String,Any}
Float = Float32

mutable struct NodeStats
  Qsa::Vector{Float}  # Q(s,a) q values 
  Nsa::Vector{Int32}  # N(s,a) visit counts
  Psa::Vector{Float}  # P(s,a) policy returned by nn
  Ns::Int64             # N(s) state visit counts
  Valids::BitVector     # Valid moves
end
MCTSStorage = Dict{State, NodeStats}

# This returns action probabilities for a state
# It performs mcts on the way down and thus changes storage
function get_action_prob!(hparams::HParams, storage::MCTSStorage, model, canonical_state::State, temp::Float)::Vector{Float}
  for i in 1:hparams["num_mcts_sims"]
    search!(hparams, storage, model, canonical_state, Vector{BitVector}())
  end

  s = Game.get_compressed_representation(canonical_state)
  node = storage[s]

  if temp == 0
    # Choose the action with the highest visitation count
    # If there are multiple, choose one of them at random
    max_c = max(node.Nsa)
    action = rand(findall(max_c .== node.Nsa))
    probs = zeros(Game.get_action_size())
    probs[action] = 1.0
    probs
  else
    # Otherwise, counts exponentiated by a temperature are proportional to probabilities
    counts = [x ^ (1.0 / temp) for x in node.Nsa]
    sum_c = sum(counts)
    [x / sum_c for x in counts]
  end
end

# Performs a MCTS search by recursive invocation
# Recursion ends as either a leaf node or a terminal state is reached
# Also, it passes down a history of states to detect infinite loops
function search!(hparams::HParams, storage::MCTSStorage, model, canonical_state::State, history::Vector{State})::Float
  s = Game.get_compressed_representation(canonical_state)

  # Check for infinite loops
  history_counts = [(count(==(s), history)) for s in unique(history)]
  # An infinite loop gets a small penalty
  if any(10 .<= history_counts)
    @info "Infinite loop reached after $(size(history, 1)) moves"
    return 1e4
  end
  
  if !haskey(storage, s)
    # We have reached a leaf, expand it and end this recursion here

    # We always check as player 1 and invert the gamestate every time
    score = Game.get_game_ended(canonical_state, 1)
    if score != 0
      # terminal
      @info "Terminal reached with score $(score)"
      return -score
    end

    pi, value = model(canonical_state)
    valids = Game.get_valid_moves(canonical_state, 1)

    # Mask the pi vector to only valid moves
    pi = pi .* valids
    sum_pi = sum(pi)
    if sum_pi <= 0
      # All valid moves had zero or lower probability whith means there is a problem with the NN 
      @warn "All $(sum(valids)) valid actions had zero or lower probability"
      pi = valids ./ sum(valids)
    else
      pi = pi ./ sum_pi
    end

    storage[s] = NodeStats(
      zeros(Game.get_action_size()),
      zeros(Game.get_action_size()),
      pi,
      0,
      valids
    )

    return -value
  end

  # Not a leaf node
  node = storage[s]
  valids = node.Valids

  # Calculate a move probability vector
  u = fill(-Inf, Game.get_action_size())
  for i in 1:Game.get_action_size()
    !node.Valids[i] && continue
    # If node was evaluated before, use the Q value
    # Otherwise use the raw formula
    u[i] = if node.Nsa[i] != 0
      hparams["cpuct"] * node.Psa[i] * sqrt(node.Ns / (1.0 + node.Nsa[i])) + node.Qsa[i]
    else
      hparams["cpuct"] * node.Psa[i] * sqrt(node.Ns + 1e-8)
    end
  end

  action = falses(Game.get_action_size())
  action_idx = argmax(u)
  action[action_idx] = 1

  
  next_state = Game.get_next_state(canonical_state, 1, action)
  next_state = Game.get_canonical_form(next_state, -1)

  # Invoke recursively
  value = search!(hparams, storage, model, next_state, vcat(history, [canonical_state]))

  # Update node
  if node.Nsa[action_idx] != 0
    node.Qsa[action_idx] = (node.Nsa[action_idx] * node.Qsa[action_idx] + value) / (node.Nsa[action_idx] + 1)
    node.Nsa[action_idx] += 1
  else
    node.Qsa[action_idx] = value
    node.Nsa[action_idx] = 1
  end
  node.Ns += 1

  -value
end

end