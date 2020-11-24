module Model

using Flux
using StatsBase
using BSON: @save, @load
using ..TakEnv
using ..Encoder

export construct_model, run_batch, train!, ModelStruct

Float = Float32

struct ModelStruct
  processing::Dense
  valuehead::Dense
  logitshead::Dense
  ModelStruct(hparams) = new(
    Dense(board_encoding_length+length(instances(Player)), hparams["hidden_size"], leakyrelu),
    Dense(hparams["hidden_size"], 1),
    Dense(hparams["hidden_size"], action_onehot_encoding_length)
  )
end

# Forward
function (model::ModelStruct)(x)
  x1 = model.processing(x)
  (model.logitshead(x1), model.valuehead(x1))
end

# Returns all trainable parameters
Flux.trainable(model::ModelStruct) = (model.valuehead, model.logitshead, model.processing)


function construct_model(hparams)
  println("Constructing model: ($(board_encoding_length), 2) -> $(hparams["hidden_size"]) -> ($(action_onehot_encoding_length), 1)")
  
  ModelStruct(hparams) |> gpu
end

function save_model(file, model)
  @save file model
end

function load_model(file)
  @load file
end

function run_batch(model, state_batch::Vector{CompressedBoard}, player_batch::Vector{Player})
  boards = Flux.batch(board_to_enc.(decompress_board.(state_batch)))
  players = Flux.onehotbatch(player_batch, instances(Player))
  
  data = vcat(players, boards) |> gpu
  logits, values = model(data)
  (cpu(logits), cpu(values))
end

function train!(hparams, model, replay_buffer)
  opt = ADAMW(1e-4)
  ps = Flux.params(model)

  logged_valueloss = []
  logged_probloss = []

  for _ in 1:hparams["train_rounds"]
    batch = rand(replay_buffer, hparams["train_batch_size"])
    states, players, probs, possible_actions, values = getindex.(batch, 1), getindex.(batch, 2), getindex.(batch, 3), getindex.(batch, 4), getindex.(batch, 5)
    states = Flux.batch(board_to_enc.(decompress_board.(states)))
    players = Flux.onehotbatch(players, instances(Player))
    inputs = vcat(players, states) |> gpu
    probs = convert.(Float, Flux.batch(map(x -> expand_action_probs(x...), zip(possible_actions, probs)))) |> gpu
    values = Flux.batch(values) |> gpu

    

    grads = Flux.gradient(ps) do
      logits_net, values_net = model(inputs)
      loss_values = Flux.Losses.mse(values_net, values)
      loss_probs = Flux.Losses.logitcrossentropy(logits_net, probs)
      loss_total = loss_values + loss_probs
      
      # Log the losses
      Flux.Zygote.ignore() do 
        push!(logged_valueloss, loss_values)
        push!(logged_probloss, loss_probs)
      end

      loss_total
    end

    Flux.update!(opt, ps, grads)
  end

  mean(logged_valueloss), mean(logged_probloss)
end

end
