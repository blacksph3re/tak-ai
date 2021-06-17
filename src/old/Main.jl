

module Main

using ..TakEnv
using ..TakEnv: black, white
using ..Encoder
using ..MCTS
using ..Model
using Distributions
using StatsBase


export play_game, train_loop, play_tournament


function play_game(hparams, storage::MCTSStorage, model)
  buf, res, storages = play_game(hparams, (storage, storage), (model, model))
  buf, res, storages[1]
end

function play_game(hparams, storages::NTuple{2, MCTSStorage}, models::NTuple{2, Any})
  board = empty_board()
  cur_player = white::Player
  first_player = white::Player
  history = []
  
  result = nothing
  first_player_result = nothing
  # Play a game
  while isnothing(result)
      storage_idx = cur_player==first_player ? 1 : 2
      
      # Perform a bit of MCTS and then choose an action
      possible_actions = compress_action.(enumerate_actions(board, cur_player))
      state = compress_board(board)
      storage = mcts_search(hparams, storages[storage_idx], state, cur_player, models[storage_idx])
      tau = hparams["steps_before_tau_0"] > length(history) ? hparams["tau"] : 0
      probs, avg_val, actions = evaluate_node(storages[storage_idx], state, tau, possible_actions)
      action_idx = sample(1:length(actions), AnalyticWeights(probs))
      push!(history, (state, cur_player, probs, actions))
      
      # Advance the game-state and check for a win
      apply_action!(board, decompress_action(actions[action_idx]), cur_player)
      result = check_win(board, cur_player)
      stalemate = check_stalemate([h[1] for h in history])
      if !isnothing(result)
          if result[1] == TakEnv.draw
              result = 0
              first_player_result = 0
          elseif result[2] == first_player
              result = 1
              first_player_result = 1
          else
              result = 1
              first_player_result = -1
          end
          break
      end
      if stalemate
        result = 0
        first_player_result = 0
        break
      end
      cur_player = opponent_player(cur_player)
      #println("Step $(length(history)) with value $(avg_val[action_idx])")
  end
  
  # Create the replay buffer by backing up the result
  replay_buffer = []
  for (state, player, probs, actions) in reverse(history)
      push!(replay_buffer, (state, player, probs, actions, result))
      result = -result
  end
  
  replay_buffer, first_player_result, storages
end

function play_tournament(hparams, models::NTuple{2, Any})
  results = [0, 0]
  
  fake_hparams = deepcopy(hparams)
  fake_hparams["tau"] = 0
  
  models_bound = map(model -> (states, players)->run_batch(model, states, players), models)
  
  for i in 1:hparams["tournament_rounds"]
      _, r, _ = play_game(fake_hparams, (MCTSStorage(), MCTSStorage()), i%2 == 0 ? models_bound : reverse(models_bound))
      
      if r > 0 && i%2 == 0 || r < 0 && i%2 != 0
          results[1] += 1
      elseif r > 0 && i%2 == 0 || r < 0 && i%2 == 0 
          results[2] += 1
      end
  end
  results
end

function train_loop(hparams)
  model = construct_model(hparams)
  target_model = deepcopy(model)
  run_batch_bound = (states, players)->run_batch(target_model, states, players)

  storage = MCTSStorage()
  replay_buffer = []
  exchange_count = 0
  
  println("Starting self-play")
  for epoch in 1:hparams["epochs"]
      start = time_ns()
      local_replay, res, storage = play_game(hparams, storage, run_batch_bound)
      replay_buffer = vcat(replay_buffer, local_replay)
      
      postgame = time_ns()
      
      if length(replay_buffer) < hparams["min_replay_buffer_length"]
          println("Still warming up, currently $(length(replay_buffer)) items in the buffer ($(length(local_replay)/(postgame-start)*1e9)it/s)")
          continue
      end
      
      valueloss, probloss = Model.train!(hparams, model, replay_buffer)
      
      posttrain = time_ns()
      
      println("Epoch $(epoch), V-loss: $(valueloss), P-loss: $(probloss), Play-time: $((postgame-start)*1e-9)s Train-time: $((posttrain-postgame)*1e-9)s")
      
      
      if epoch>0 && epoch%hparams["play_tournament_every"]==0
          println("Playing tournament...")
          res = play_tournament(hparams, (model, target_model))
          
          # Check if we have to exchange the model
          if res[1] >= sum(res)*hparams["model_exchange_threshold"]
              println("Model tournament won ($(res[1]) - $(res[2])), exchanging model $(exchange_count) and checkpointing")
              replay_buffer = []
              storage = MCTSStorage()
              target_model = deepcopy(model)
              run_batch_bound = (states, players)->run_batch(target_model, states, players)
              exchange_count += 1
              save_model("model_$(exchange_count).bson", model)
          else
            println("No winner yet - score was $(res[1]) - $(res[2])")
          end
      end
  end
  
end

end