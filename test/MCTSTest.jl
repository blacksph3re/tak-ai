include("../src/MCTS.jl")

module MCTSTest
using ..TakEnv
using ..TakEnv: black, white
using ..Encoder
using ..MCTS
using Test
using Distributions
Float = Float32

@testset "MCTS.jl" begin
  hparams = Dict(
    "d_puct" => 1.0,
    "mcts_batch_size" => 64,
    "mcts_iterations" => 20,
    "run_name" => "testrun",
    "exploration_factor" => 0.25,
    "separate_models" => false, # Whether to use separate models for playing as white or as black
  )
  HParams = typeof(hparams)

  function dummy_model(state, player)
    logits = rand(Dirichlet(action_onehot_encoding_length, 0.1), length(state))
    values = rand(length(state))
    return logits, values
  end

  @testset "perform a search" begin
    storage = MCTSStorage()
    storage = mcts_search(hparams, storage, compress_board(empty_board()), white::Player, dummy_model)
    @test length(storage) > 20

    summed = 0
    for x in storage
      stats = x[2]
      summed += sum(stats.visit_counts)

      @test length(stats.actions) == length(stats.visit_counts) == length(stats.values) == length(stats.avg_values) == length(stats.probs)
      @test isapprox(sum(stats.probs), 1)
    end
    @test summed > length(storage)
  end

  @testset "merge storages" begin
    a = MCTSStorage()
    a = mcts_search(hparams, a, compress_board(empty_board()), white::Player, dummy_model)
    b = MCTSStorage()
    b = mcts_search(hparams, b, compress_board(empty_board()), white::Player, dummy_model)

    @test a != b
    summed_a = sum((sum(s.visit_counts) for (_, s) in a))
    summed_b = sum((sum(s.visit_counts) for (_, s) in b))

    MCTS.merge_storages!(a, b)

    @test summed_a + summed_b == sum((sum(s.visit_counts) for (_, s) in a))
  end

end
end