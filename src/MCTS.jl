module MCTS

using ..TakEnv
using ..Encoder
using Distributions
using NNlib

export MCTSStorage, search_batch, mcts_search

HParams = Dict{String,Any}
Float = Float64

mutable struct NodeStats
  actions::Vector{CompressedAction}
  visit_counts::Vector{Int}
  values::Vector{Float}
  avg_values::Vector{Float}
  probs::Vector{Float}
end

MCTSStorage = Dict{CompressedBoard, NodeStats}


function is_leaf(storage::MCTSStorage, state::CompressedBoard)
  !haskey(storage, state)
end

function find_leaf(hparams::HParams, storage::MCTSStorage, start_state::CompressedBoard, player::Player)
  states = CompressedBoard[]
  actions = CompressedAction[]
  
  cur_state = start_state
  cur_player = player
  value = nothing
  
  while !is_leaf(storage, cur_state)
      push!(states, cur_state)
      
      stats = storage[cur_state]
      total_sqrt = convert(Float, sqrt(sum(stats.visit_counts)))
      probs = stats.probs
      
      # For the first move, add noise for exploration
      if cur_state == start_state
          noise_dist = Dirichlet(length(stats.actions), 0.03)
          exploration = convert(Float, hparams["exploration_factor"])
          probs = (1-exploration) .* stats.probs .+ exploration .* rand(noise_dist)
      end
      
      # Calculate action scores
      d_puct = convert(Float, hparams["d_puct"])
      score = stats.avg_values .+ d_puct .* probs .* total_sqrt ./ (1 .+ stats.visit_counts)
      
      
      # Advance the game state
      board = decompress_board(cur_state)
      possible_actions = enumerate_actions(board, cur_player)
      possible_actions_compressed = compress_action.(possible_actions)
      
      # Set all those score values to -Inf which are invalid actions
      for (i, a) in enumerate(stats.actions)
        if !(a in possible_actions_compressed)
          score[i] = -Inf
        end
      end
      
      # Choose the argmax action
      cur_action = stats.actions[argmax(score)]
      push!(actions, cur_action)
      apply_action!(board, decompress_action(cur_action), cur_player)  
      
      # Check for win
      result = check_win(board, player)
      cur_player = TakEnv.opponent_player(cur_player)
      cur_state = compress_board(board)
      
      # We already switched players, so a victory was actually a loss
      if !isnothing(result)
          victory_type, player_won = result
          if victory_type == TakEnv.draw::ResultType
              value = 0.0
          elseif player_won == cur_player
              value = -1.0
          else
              value = 1.0
          end
          break
      end
  end
  
  value, cur_state, cur_player, states, actions
end

function search_batch(hparams::HParams, storage::MCTSStorage, start_state, player, model)::MCTSStorage
  lock_win = Threads.SpinLock()
  lock_expand = Threads.SpinLock()
  backup_queue = []
  expand_states = []
  expand_players = []
  expand_queue = []
  planned = []
  
  # Perform a search
#     Threads.@threads for _ in 1:hparams["batch_size"]
  for _ in 1:hparams["batch_size"]
      value, leaf_state, leaf_player, states, actions = find_leaf(hparams, storage, start_state, player)
      if !isnothing(value) # Win/lose/draw
          lock(lock_win)
          try
              push!(backup_queue, (value, states, actions))
          finally
              unlock(lock_win)
          end
      elseif !(leaf_state in planned) # Need to expand
          # For expansion, precalculate all possible actions
          possible_actions = compress_action.(enumerate_actions(decompress_board(leaf_state)))
          
          lock(lock_expand)
          try
              push!(planned, leaf_state)
              push!(expand_states, leaf_state)
              push!(expand_players, leaf_player)
              push!(expand_queue, (leaf_state, states, actions, possible_actions))
          finally
              unlock(lock_expand)
          end
      end
  end
  
  # Expand nodes
  if length(expand_queue)>0
      # Get values and logits via model
      logits, values = model(expand_states, expand_players)
      
      # Save the node
      for ((leaf_state, states, actions, possible_actions), logit, value) in zip(expand_queue, eachrow(logits), values)          
          stats = NodeStats(
              possible_actions,
              zeros(length(possible_actions)), # Visit counts
              zeros(length(possible_actions)), # Values
              zeros(length(possible_actions)), # Avg values
              softmax(logit[possible_actions .+ 1]) # Probs, softmaxed across all possible actions (we're using the fact that the possible action compression is a number between 0 and num(actions))
          )
          storage[leaf_state] = stats
          push!(backup_queue, (value, states, actions))
      end
  end
  
  for (value, states, actions) in backup_queue
      cur_value = -value
      for (s, a) in zip(reverse(states), reverse(actions))
          prev_sum = sum(storage[s].visit_counts)
          stats = storage[s]
          act_idx = findfirst(x->x==a, stats.actions)
          stats.visit_counts[act_idx] += 1
          stats.values[act_idx] += cur_value
          stats.avg_values[act_idx] = stats.values[act_idx] / stats.visit_counts[act_idx]
          cur_value = -cur_value
          @assert sum(storage[s].visit_counts) == prev_sum+1
      end
  end
  storage
end

function mcts_search(hparams::HParams, storage::MCTSStorage, start_state, player, model)::MCTSStorage
  for _ in 1:hparams["mcts_iterations"]
    storage = search_batch(hparams, storage, start_state, player, model)
  end
  storage
end

end