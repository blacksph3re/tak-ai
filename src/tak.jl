module tak

include("TakEnv.jl")
include("Encoder.jl")
include("MCTS.jl")
include("Model.jl")
include("Main.jl")

function main()
  hparams = Dict(
    "d_puct" => 1.0,
    "mcts_batch_size" => 16,
    "mcts_iterations" => 10,
    "run_name" => "testrun",
    "exploration_factor" => 0.25,
    "hidden_size" => 2048,
     # Tak has a white-bias (meaning the starting player is 60% more likely to win) - 
    # starting player randomization could balance this out
    "randomize_starting_player" => false,
    # How strong it should add random to move selection (0 -> fully greedy, inf -> fully random)
    "tau" => 1.0,
    # How many steps to apply above tau before switching to fully greedy
    "steps_before_tau_0" => 8,
    # How long to self-play before doing a training run
    "min_replay_buffer_length" => 1000,
    # How many rounds of training per game
    "train_rounds" => 10,
    # Training batch size
    "train_batch_size" => 256,
    "epochs" => 1000000,
    # How many rounds to play a tournament
    "tournament_rounds" => 50,
    # Every x epochs to play a tournament
    "play_tournament_every" => 50,
    # The threshold which results in model exchange
    # In fraction of games won
    "model_exchange_threshold" => 0.6
  )

  Main.train_loop(hparams)
end
end