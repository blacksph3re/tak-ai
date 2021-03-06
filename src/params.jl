#####
##### Training hyperparameters
#####

using AlphaZero

# Network = NetLib.SimpleNet

# netparams = NetLib.SimpleNetHP(
#   width=64,
#   depth_common=6,
#   use_batch_norm=true,
#   batch_norm_momentum=1.)

Network = NetLib.ResNet 

netparams = NetLib.ResNetHP(
  num_filters=256,
  num_blocks=10,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=32,
  num_value_head_filters=32,
  batch_norm_momentum=0.1
)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=5000,
    num_workers=8,
    batch_size=8,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=600,
    cpuct=2.0,
    prior_temperature=1.0,
    temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.3]),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=128,
    num_workers=8,
    batch_size=8,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.5,
    alternate_colors=true),
  mcts=MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.05)

learning = LearningParams(
  use_gpu=true,
  use_position_averaging=true,
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=1024,
  loss_computation_batch_size=1024,
  optimiser=Adam(lr=2e-3),
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2000,
  num_checkpoints=1)

mem_analysis = MemAnalysisParams(
  10
)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=15,
  ternary_rewards=true,
  use_symmetries=true,
  memory_analysis=mem_analysis,
  mem_buffer_size=PLSchedule(
  [      0,        15],
  [400_000, 1_000_000]))

#####
##### Evaluation benchmark
#####

mcts_baseline =
  Benchmark.MctsRollouts(
    MctsParams(
      arena.mcts,
      num_iters_per_turn=1000,
      cpuct=1.))

# minmax_baseline = Benchmark.MinMaxTS(
#   depth=5,
#   τ=0.2,
#   amplify_rewards=true)

alphazero_player = Benchmark.Full(arena.mcts)

network_player = Benchmark.NetworkOnly(τ=0.5)

benchmark_sim = SimParams(
  arena.sim;
  num_games=64,
  num_workers=8,
  batch_size=8,
  alternate_colors=false)

benchmark = [
  Benchmark.Duel(alphazero_player, mcts_baseline,   benchmark_sim),
# Benchmark.Duel(alphazero_player, minmax_baseline, benchmark_sim),
  Benchmark.Duel(network_player,   mcts_baseline,   benchmark_sim),
# Benchmark.Duel(network_player,   minmax_baseline, benchmark_sim)
]

#####
##### Wrapping up in an experiment
#####

experiment = Experiment("tak", TakInterface.TakSpec(), params, Network, netparams, benchmark)