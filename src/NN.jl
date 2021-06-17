module NN

using Flux
using StatsBase
using BSON: @save, load

HParams = Dict{String, Any}
Float = Float32

struct ModelStruct
  layer1::Dense
  layer2::Dense
  logitshead::Dense
  valuehead::Dense
  ModelStruct(hparams, in_size, out_size) = new(
    Dense(in_size, hparams["hidden_size"], leakyrelu) |> gpu,
    Dense(hparams["hidden_size"], hparams["hidden_size"], leakyrelu) |> gpu,
    Dense(hparams["hidden_size"], out_size) |> gpu,
    Dense(hparams["hidden_size"], 1, tanh) |> gpu,
  )
end

# Forward
function (model::ModelStruct)(x)
  x = model.layer1(x)
  x = model.layer2(x)
  (model.logitshead(x), model.valuehead(x))
end

# Returns all trainable parameters
Flux.trainable(model::ModelStruct) = (model.layer1, model.layer2, model.logitshead, model.valuehead)


function init(hparams::HParams, state_encoding_length::Int, action_encoding_length::Int)::ModelStruct
  model = ModelStruct(hparams, state_encoding_length, action_encoding_length) |> gpu
  model.layer1.b .= randn(hparams["hidden_size"]) .* 0.1
  model
end

# Get pi, v for a state
function predict(model::ModelStruct, state::BitVector)::Tuple{Vector{Float}, Float}
  logits, value = model(state |> gpu)
  (logits |> softmax |> cpu, (value |> cpu)[1])
end

# Train on a list of (state, pi, v) from MCTS
function train!(hparams::HParams, model::ModelStruct, data)
  opt = Flux.Optimise.RADAM(hparams["lr"])
  ps = Flux.params(model)

  loader = Flux.Data.DataLoader((getindex.(data, 1), getindex.(data, 2), getindex.(data, 3)), batchsize=hparams["batch_size"], shuffle=true)

  for epoch in 1:hparams["training_loops"]
    losses = []
    for (states, pi, v) in loader
      states = gpu(Flux.stack(states, 2))
      pi = gpu(Flux.stack(pi, 2))
      v = gpu(Flux.unsqueeze(v, 2))

      loss, back = Flux.Zygote.pullback(ps) do
        pi_pred, v_pred = model(states)
        Flux.Losses.logitcrossentropy(pi_pred, pi) + Flux.Losses.mse(v_pred, v)
      end

      gs = back(one(loss))

      Flux.Optimise.update!(opt, ps, gs)

      push!(losses, cpu(loss))
    end

    mean_loss = sum(losses) / size(losses, 1)
    @info "Epoch $(epoch) Loss $(mean_loss) in $(size(losses, 1)) iterations"
  end
end

function save_model(file, model)
  @save file model
end

function load_model(file)
  load(file, @__MODULE__)[:model]
end

end