# This interface implements the functions from https://github.com/suragnair/alpha-zero-general
# Potentially, this code could then be linked up into that repository or other games can be used instead

module TakInterface

using ..TakEnv
using ..Encoder

export Board, Action, CompressedBoard, get_init_board, get_board_size, get_action_size, get_next_state, get_valid_moves, get_game_ended, get_canonical_form, get_symmetries, get_compressed_representation

Board = Encoder.BoardEnc
Action = Encoder.ActionEnc
CompressedBoard = Encoder.CompressedBoard

# Returns an empty board, encoded
function get_init_board()::Board
  Encoder.board_to_enc(TakEnv.empty_board())
end

# Returns board dimensions
function get_board_size()::Int
  Encoder.board_encoding_length
end

# Number of possible actions
function get_action_size()::Int
  Encoder.action_onehot_encoding_length
end

# Advances the gamestate by one
function get_next_state(board::Board, player::Int, action::Action)::Board
  board = Encoder.enc_to_board(board)
  action = Encoder.onehot_to_action(action)
  player = player == 1 ? TakEnv.white : TakEnv.black
  TakEnv.apply_action!(board, action, player)

  Encoder.board_to_enc(board)
end

# Returns an action vector with a bit set for every allowed action
function get_valid_moves(board::Board, player::Int)::BitVector
  board = Encoder.enc_to_board(board)
  player = player == 1 ? TakEnv.white : TakEnv.black

  possible_actions = Encoder.action_to_onehot.(TakEnv.enumerate_actions(board, player))

  # Reduce one-hot encoded actions with bitwise or
  reduce(.|, possible_actions, init=falses(get_action_size()))
end

# Returns a positive score if the current player won
# Also checks for infinite loops
function get_game_ended(board::Board, player::Int)::Float64
  board = Encoder.enc_to_board(board)
  player = player == 1 ? TakEnv.white : TakEnv.black

  res = TakEnv.check_win(board, player)
  if isnothing(res)
    return 0
  end

  restype, winner = res

  if restype == TakEnv.draw
    # Draws are not cool
    return 1e-4
  else
    score = 1.0
    # Road wins are extra cool
    if restype == TakEnv.flat_win
      score = 0.9
    end

    if winner != player
      score *= -1
    end

    return score
  end
end

# Returns the canonical form of the board with the view from the white player
# Inverts the board if black is active
function get_canonical_form(board::Board, player::Int)::Board
  if player == 1
    board
  else
    Encoder.board_to_enc(TakEnv.invert_board(Encoder.enc_to_board(board)))
  end
end

# Returns all 8 symmetries of a gamestate
function get_symmetries(board::Board, pi::AbstractArray)::Vector{Tuple{Board, AbstractArray}}
  symmetries = [(copy(board), copy(pi))]
  board = Encoder.enc_to_board(board)
  push!(symmetries, (Encoder.board_to_enc(TakEnv.mirror_board(board)), Encoder.mirror_action_vec(pi)))
  for _ in 1:3
    board = TakEnv.rotate_board(board)
    pi = Encoder.rotate_action_vec(pi)
    push!(symmetries, (Encoder.board_to_enc(board), pi))
    push!(symmetries, (Encoder.board_to_enc(TakEnv.mirror_board(board)),
                        Encoder.mirror_action_vec(pi)))
  end
  unique(symmetries)
end

# Return a compressed representation for storage in the map
function get_compressed_representation(board::Board)::CompressedBoard
  Encoder.compress_board(Encoder.enc_to_board(board))
end

end