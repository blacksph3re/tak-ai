module tak

include("TakEnv.jl")
include("Encoder.jl")

using .TakEnv
using .Encoder

function main()
  render_board(random_game()[1])

  print("huhu")
end
end