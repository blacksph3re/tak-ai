

module TakInterface

using Base: Float32
import AlphaZero.GI

using Crayons

using ..TakEnv
using ..Encoder
using ..TakUI

const COMPRESSION_ACTIVE = true

compress_board = COMPRESSION_ACTIVE ? Encoder.compress_board : identity
decompress_board = COMPRESSION_ACTIVE ? Encoder.decompress_board : identity
StateType = COMPRESSION_ACTIVE ? Tuple{Encoder.CompressedBoard, TakEnv.Player, UInt16} : Tuple{Encoder.BoardEnc, TakEnv.Player, UInt16}

struct TakSpec <: GI.AbstractGameSpec end

mutable struct TakGame <: GI.AbstractGameEnv
  board :: TakEnv.Board
  curplayer :: TakEnv.Player
  result :: Union{Nothing, TakEnv.Result}
  timer :: UInt16
end

GI.spec(::TakGame) = TakSpec()
GI.two_players(::TakSpec) = true
GI.actions(::TakSpec) = collect(1:Encoder.action_onehot_encoding_length)
GI.state_type(::TakSpec) = StateType

function GI.vectorize_state(::TakSpec, state)::Array{Float32}
  board, player, _ = state
  board = decompress_board(board)
  if player == TakEnv.black::Player
    board = TakEnv.invert_board(board)
  end
  convert.(Float32, Encoder.board_to_enc(board))
end

function GI.init(::TakSpec)::TakGame
  TakGame(
    TakEnv.empty_board(),
    TakEnv.white::Player,
    nothing,
    0
  )
end

function GI.set_state!(g::TakGame, s)
  g.board = decompress_board(s[1])
  g.curplayer = s[2]
  g.result = TakEnv.check_win(g.board, g.curplayer)
  g.timer = s[3]
end

function GI.current_state(g::TakGame)::StateType
  (Encoder.compress_board(g.board), g.curplayer, g.timer)
end

function GI.game_terminated(g::TakGame)::Bool
  !isnothing(g.result)
end

function GI.white_playing(g::TakGame)::Bool
  g.curplayer == TakEnv.white::Player
end

function GI.actions_mask(g::TakGame)::BitVector
  if !isnothing(g.result)
    falses(Encoder.action_onehot_encoding_length)
  else
    Encoder.get_valid_moves(g.board, g.curplayer)
  end
end

function GI.play!(g::TakGame, action_idx)
  action = falses(Encoder.action_onehot_encoding_length)
  action[action_idx] = 1

  TakEnv.apply_action!(g.board, Encoder.onehot_to_action(action), g.curplayer)
  g.result = TakEnv.check_win(g.board, g.curplayer)
  g.timer += 1
  if g.timer >= 500
    g.result = (TakEnv.stalemate::ResultType, nothing)
  end

  if isnothing(g.result)
    g.curplayer = TakEnv.opponent_player(g.curplayer)
  end

end

const SCORE_TABLE = Dict{TakEnv.ResultType, Float64}(
  TakEnv.flat_win::ResultType => 1,
  TakEnv.road_win::ResultType => 1,
  TakEnv.double_road_win::ResultType => 1,
  TakEnv.draw::ResultType => 0,
  TakEnv.stalemate::ResultType => 0,
)

function GI.white_reward(g::TakGame)::Float64
  if isnothing(g.result)
    0
  else
    score = SCORE_TABLE[g.result[1]]
    if g.result[2] === TakEnv.black::Player
      score *= -1
    end
    score
  end
end

# TODO redefine
function GI.heuristic_value(g::TakGame)::Float64
  0.0
end

function GI.symmetries(::TakSpec, state)::Array{Tuple{StateType, Vector{Int}}}
  board, player, timer = state
  actions = collect(1:Encoder.action_onehot_encoding_length)

  symmetries = [((board, player, timer), actions)]

  board = decompress_board(board)
  push!(symmetries, ((compress_board(TakEnv.mirror_board(board)), player, timer), Encoder.mirror_action_vec(actions)))
  for _ in 1:3
    board = TakEnv.rotate_board(board)
    actions = Encoder.rotate_action_vec(actions)
    push!(symmetries, ((compress_board(board), player, timer), actions))
    push!(symmetries, ((compress_board(TakEnv.mirror_board(board)), player, timer), Encoder.mirror_action_vec(actions)))
  end
  unique(symmetries)
end

function GI.action_string(::TakSpec, action_idx)::String
  action = falses(Encoder.action_onehot_encoding_length)
  action[action_idx] = true
  action = Encoder.onehot_to_action(action)

  TakUI.action_string(action)
end

function GI.parse_action(::TakSpec, action_input::String)
  try
    action = TakUI.read_action(action_input)
    findfirst(Encoder.action_to_onehot(action))
  catch
    nothing
  end
end

function GI.render(g::TakGame)
  print("Current player: ", TakUI.player_color(g.curplayer), g.curplayer == TakEnv.white ? "white" : "black", "\n")

  print(TakUI.render_board_cmd(g.board))
end

end