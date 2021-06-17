include("../src/Model.jl")

module ModelTest

using ..TakEnv
using ..Encoder
using ..Model
using Test

@testset "ModelTest.jl" begin
  @testset "empty board through model" begin
    hparams = Dict(
      "hidden_size" => 2048,
    )

    model = construct_model(hparams)

    states = compress_board.(fill(empty_board(), 10))
    players = fill(white::Player, 10)

    logits, values = run_batch(model, states, players)

    @test size(values) == (1, 10)
    @test size(logits) == (action_onehot_encoding_length, 10)

  end

end
end