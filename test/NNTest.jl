
module NNTest

using ..NN

using Test
using Flux

@testset "NN.jl" begin
  hparams = Dict(
    "state_size" => 64,
    "action_size" => 32,
    "hidden_size" => 128,
    "training_loops" => 2,
    "batch_size" => 4,
    "lr" => 1e-4,
    "checkpoint_dir" => "./checkpoints"
  )

  @testset "construct model, predict" begin
    model_data = NN.init(hparams)
    model = (in) -> NN.predict(model_data, in)

    in = falses(64)

    pi, v = model(in)
    @test size(v, 1) == 1
    v = v[1]
    @test v >= -1
    @test v <= 1
    @test isapprox(sum(pi), 1)

  end

  @testset "training run" begin
    model = NN.init(hparams)

    function randinput()
      x = falses(64)
      x[rand(1:64)] = 1
      x
    end

    randdata = [(randinput(), softmax(randn(32)), tanh(randn())) for _ in 1:100]
    old_model = deepcopy(model)

    @test old_model.layer1.W == model.layer1.W
    NN.train!(hparams, model, randdata)
    @test old_model.layer1.W != model.layer1.W

  end

  @testset "saving/loading" begin
    model = NN.init(hparams)

    NN.save_model("/tmp/tmp-tak-model.bson", model)
    new_model = NN.load_model("/tmp/tmp-tak-model.bson")
    @test model.layer1.W == new_model.layer1.W
  end

end

end