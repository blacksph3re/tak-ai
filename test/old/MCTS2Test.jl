module MCTSTest

using ..MCTS
using ..TakInterface
using ..NN

using Test

@testset "MCTS.jl" begin
  hparams = Dict{String, Any}(
    "cpuct" => 0.9,
    "num_mcts_sims" => 1e6,
    "hidden_size" => 16,
  )

  model_data = NN.init(hparams, TakInterface.get_board_size(), TakInterface.get_action_size())
  model = (in) -> NN.predict(model_data, in)

  @testset "perform a search" begin
    mcts = MCTS.MCTSStorage()
    state = TakInterface.get_canonical_form(TakInterface.get_init_board(), 1)

    MCTS.get_action_prob!(hparams, mcts, model, state, 0.1f0)

    @test haskey(mcts, TakInterface.get_compressed_representation(state))
    @test size(collect(keys(mcts)), 1) <= hparams["num_mcts_sims"]
    @info size(collect(keys(mcts)))
  end
end

end