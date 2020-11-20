# ----------------------------------------------------------
# Thought about board representations to the NN:

# The tak board size is 5x5 (6x6) holding 6 possible stones, but there is a third dimension
# In theory, stones can be stacked to any height - maximum all normal stones plus 1 capstone
# Thus, a 5x5 board becomes a 5x5x43*6 board = 6450 board combinations
# With bigger boards, this representation quickly becomes intractable (6x6x61x6=13176, 7x7 -> 23813, 38784)
# 
# Also, this representation is relatively sparse, as stacks in practice never reach this height
# It doesn't make sense from a strategic perspective to stack thia high
#
# Next, only flat stones can be stacked upon, so there are only 2 options for within a stack - white/black
# Resultingly the complexity is 5*5*6+5*5*42*2 = 2250 (4536, 8134, 13184)
# 
# Lastly, the topmost stones have special significance, as they are those responsible for forming roads
# Thus a board representation should have a fixed point in the input for the topmost stones

# Idea 1: Represent it as a (5x5x6) x 43 board and use 1D convolutions
# Idea 2: Reverse gravity - instead of having the board at position [1], put the top at position [end] and stack under that
# Idea 3: Use an attention mechanism to attend to the varying-length stacks
# Idea 4: Chop off the stack after a certain height (e.g. 2*FIELD_SIZE)
# Idea 4.1: Chop off the stack but display some information about chopped off items, e.g. 
# Idea 5: Have different representations for top stones and stacked stones

# A combination of 4, 5 and 2 might make it viable for a fully connected net.
# If we e.g. chop off height>3*FIELD_SIZE with an additional bit that there is still something left, that would be
# 5*5*6+5*5*15*2+5*5 = 925 (1548, 2401, 3520)
# These are still within the reach of a fully connected layer

# -------------------------------------------------------------
# Board representation for storage
#
# For storage, we want to use as little memory as possible while having the state be perfectly compressible
# The naive way of storing the board would take 5*5*42*7=7350 bit 
# However, the stacks are usually of low height and we could use a smaller representation
# Also, as within stacks there are only flat stones, we could save some bits here 
#
# Thus, we could start with encoding the top pieces, which have 7 options (3 bit each)
# Then we could encode the stack heights (max 100 for 8*8 -> 7 bit each)
# Then we could concatenate the stacks with 1 bit per stone

# ---------------------------------------------------------
# Thoughts about action representations:

# Calculates all possible moves for a board
# To make it reinforement learning compatible, we need a fixed size action vector
# Thus, let's first enumerate all moves

# Placements can be made on every field (which is empty), 3 stone types can be placed
# -> 3x5x5 = 75

# Mocement
# A stone or entire stack can be moved 1 field in each direction
# -> 4x5x5 = 100

# Carrying
# A stack can be carried multiple fields, for which on each field it must drop at least one stone
# The carry limit is equal to the FIELD_SIZE (5)
# -> 4 Directions x number of carried stones 5
# -> 5x5x4x26 carries
# -> 5x5x4x26 = 2600

# Total:
# 7600

# -> This is not handlebar for a NN, we have to split the action
# -> We need a two-stage action, first choosing the field and then the action
# The alternative would be to split the carrying moves into several consecutive moves where the side doesn't change

# Field: 5x5 = 25

# Placements: 3
# Movements: 4
# Carrying: 4x5x4x3x2x1 = 480
# Total: 487

# ---------------------------------------------------------------

# Include should happen separately
#include("TakEnv.jl")

module Encoder

using ..TakEnv
using ..TakEnv: FIELD_SIZE, FIELD_HEIGHT, possible_carries, possible_directions, stone_player, stone_type, get_stack_height, flat, cap, stand, north, south, east, west, placement, carry, white, black

export board_encoding_length, action_onehot_encoding_length, action_twohot_encoding_length
export action_to_onehot, onehot_to_action, action_to_twohot, twohot_to_action, compress_action, decompress_action
export board_to_enc, enc_to_board, compress_board, decompress_board
export BoardEnc, ActionEnc, ActionTwoEnc, CompressedBoard, CompressedAction

BoardEnc = BitArray{1}
ActionEnc = BitArray{1}
ActionTwoEnc = Tuple{BitArray{1}, BitArray{1}}
CompressedBoard = BitArray{1}
CompressedAction = Int

const STACK_REPR_HEIGHT = FIELD_SIZE*3
@assert STACK_REPR_HEIGHT < FIELD_HEIGHT-1

const action_onehot_encoding_length = (length(possible_directions)*length(possible_carries)+length(instances(Stone)))*FIELD_SIZE^2
const action_twohot_encoding_length = (length(possible_directions)*length(possible_carries)+length(instances(Stone)), FIELD_SIZE^2)
const action_encoding_first_length = length(possible_directions)*length(possible_carries)+length(instances(Stone))

# We want a list of all possible moves
# All placement moves are easy, just looking at the fields on the board which are empty:

function pos_to_onehot(x::Integer, y::Integer)::BitArray{1}
  @assert x>=1 && x<=FIELD_SIZE
  @assert y>=1 && y<=FIELD_SIZE
  
  retval = falses(FIELD_SIZE^2)
  retval[x + (y-1)*FIELD_SIZE] = true
  return retval
end

function pos_to_onehot(p::NTuple{2, Integer})::BitArray{1}
  pos_to_onehot(p[1], p[2])
end

function pos_to_idx(x::Integer, y::Integer)::Integer
  x + (y-1)*FIELD_SIZE
end
function pos_to_idx(p::NTuple{2, Integer})::Integer
  pos_to_idx(p[1], p[2])
end

function onehot_to_pos(vec::BitArray{1})::Union{Tuple{Integer, Integer}, Nothing}
  pos = findfirst(vec)
  if isnothing(pos)
      return nothing
  end
  # Fuck 1-Termination
  x = (pos-1) % FIELD_SIZE + 1
  y = (pos-1) ÷ FIELD_SIZE + 1
  return (x, y)
end

function idx_to_pos(idx::Integer)::NTuple{2, Integer}
  x = (idx-1) % FIELD_SIZE + 1
  y = (idx-1) ÷ FIELD_SIZE + 1
  return (x, y)
end

# stone to one-hot
function stone_to_onehot(s::Stone)::BitArray{1}
  retval = falses(3)
  retval[Integer(s)+1] = true
  return retval
end

function onehot_to_stone(vec::BitArray{1})::Union{Stone, Nothing}
  pos = findfirst(vec)
  if isnothing(pos)
      return nothing
  end
  
  return Stone(pos-1)
end

function direction_to_onehot(m::Direction)::BitArray{1}
  retval = falses(4)
  retval[Integer(m)+1] = true
  return retval
end

function onehot_to_direction(vec::BitArray{1})::Union{Direction,Nothing}
  pos = findfirst(vec)
  if isnothing(pos)
      return nothing
  end
  
  return Direction(pos-1)
end

function carry_to_onehot(m::Direction, carry::CarryType)::BitArray{1}
  retval = falses(4*length(possible_carries))
  carry_idx = findfirst(c -> c==carry, possible_carries)
  
  @assert !isnothing(carry_idx) "the carry $(carry) is not a valid one"
  
  dir_idx = Integer(m)
  retval[dir_idx*length(possible_carries) + carry_idx] = true
      
  return retval
end

function onehot_to_carry(vec::BitArray{1})::Union{Tuple{Direction,CarryType}, Nothing}
  pos = findfirst(vec)
  if isnothing(pos)
      return nothing
  end
  
  carry_idx = (pos-1) % length(possible_carries) + 1
  dir_idx = (pos-1) ÷ length(possible_carries)
      
  return Direction(dir_idx), possible_carries[carry_idx]
end

function action_to_first_idx(a::Action)::Int
  if a.action_type == placement::ActionType
    Int(a.stone)
  else
    carry_idx = findfirst(c -> c==a.carry, possible_carries) - 1
    dir_idx = Int(a.dir)
    3+dir_idx*length(possible_carries)+carry_idx
  end
end

function action_to_idx(a::Action)::Int
  (pos_to_idx(a.pos) - 1) * action_encoding_first_length + action_to_first_idx(a)
end

const action_encodings_lookup = [setindex!(falses(action_onehot_encoding_length), true, i) for i in 1:action_onehot_encoding_length]::Array{ActionEnc,1}
function action_to_onehot(a::Action)::ActionEnc
  action_encodings_lookup[action_to_idx(a) + 1]
end

compress_action(a::Action)::CompressedAction = action_to_idx(a)

function first_idx_to_action(actidx::Int, pos::NTuple{2, Int})::Action
  if actidx < 3
    Action(pos, Stone(actidx), nothing, nothing, placement::ActionType)
  else
      actidx -= 3
      dir_idx = actidx ÷ length(possible_carries)
      carry_idx = actidx % length(possible_carries)

      Action(pos, nothing, Direction(dir_idx), possible_carries[carry_idx+1], carry::ActionType)
  end
end

function idx_to_action(idx::Int)::Action
  first_idx_to_action(idx % action_encoding_first_length, idx_to_pos(idx ÷ action_encoding_first_length + 1))
end

decompress_action(idx::CompressedAction)::Action = idx_to_action(idx)

function onehot_to_action(vec::ActionEnc)::Action
  idx = findfirst(vec)-1
  idx_to_action(idx)
end

function action_to_twohot(action::Action)::ActionTwoEnc
  posenc = pos_to_onehot(action.pos)
  
  actenc = if action.action_type == placement::ActionType
      vcat(stone_to_onehot(action.stone), zeros(4*length(possible_carries)))
  else
      vcat(zeros(3), carry_to_onehot(action.dir, action.carry))
  end
  
  (posenc, actenc)
end

function twohot_to_action(vec::ActionTwoEnc)::Action
  pos = onehot_to_pos(vec[1])
  if vec[2][1] || vec[2][2] || vec[2][3]
      stone = onehot_to_stone(vec[2][1:3])
      Action(pos, stone, nothing, nothing, placement::ActionType)
  else
      dir, c = onehot_to_carry(vec[2][4:end])
      Action(pos, nothing, dir, c, carry::ActionType)
  end
end


# "Float" the board - meaning the top row is at the same z and stacks are below that
function float_board(board::Board)::Board
  newboard = empty_board()
  for x in 1:FIELD_SIZE
    for y in 1:FIELD_SIZE
      if isnothing(board[x,y,1])
        continue
      end
      z2 = 0
      for z in reverse(1:TakEnv.FIELD_HEIGHT)
        if !isnothing(board[x,y,z])
          newboard[x,y,TakEnv.FIELD_HEIGHT-z2] = board[x,y,z]
          z2 += 1
        end
      end
    end
   end
   newboard
end

function unfloat_board(board::Board)::Board
   newboard = empty_board()
   for x in 1:FIELD_SIZE
       for y in 1:FIELD_SIZE
           z2 = 1
           for z in 1:TakEnv.FIELD_HEIGHT
               if !isnothing(board[x,y,z])
                   newboard[x,y,z2] = board[x,y,z]
                   z2 += 1
               end
           end
       end
   end
   newboard
end


const piece_encodings = vcat([falses(6)], [setindex!(falses(6), true, i) for i in 1:6])
function piece_to_onehot(piece::Union{Tuple{Stone, Player}, Nothing})::BitArray{1}
  if isnothing(piece)
      return piece_encodings[1]
  end
  player_offset = Int(piece[2]) * 3 # white -> 0, black -> 3
  piece_encodings[2+Int(piece[1])+player_offset]
end

function onehot_to_piece(vec::BitArray{1})::Union{Tuple{Stone, Player}, Nothing}
  @assert length(vec) == 6
  idx = findfirst(vec)
  if isnothing(idx)
      return nothing
  end
  player = Player(Int(floor((idx-1)/3)))
  stone = Stone((idx-1)%3)
  (stone, player)
end

function board_top_row_to_onehot(board::Board)::BitArray{1}
  retval = falses(6*FIELD_SIZE*FIELD_SIZE)
  offset = 1
  for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
          retval[offset:offset+5] = piece_to_onehot(board[x,y,end])
          offset += 6
      end
  end
  retval
end

function onehot_to_board_top_row!(board::Board, vec::BitArray{1})
  @assert length(vec) == 6*FIELD_SIZE*FIELD_SIZE
  offset = 1
  for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
          board[x,y,end] = onehot_to_piece(vec[offset:offset+5])
          offset += 6
      end
  end
end

const player_encodings = [setindex!(falses(2), true, i) for i in 1:2]
@inline function player_to_onehot(player::Player)::BitArray{1}
  player_encodings[Int(player)+1]
end

function onehot_to_player(vec::BitArray{1})::Union{Player, Nothing}
  if vec[1]
      white::Player
  elseif vec[2]
      black::Player
  else
      nothing
  end
end

function stacks_to_onehot(board::Board)::BitArray{1}
  retval = falses(FIELD_SIZE*FIELD_SIZE*STACK_REPR_HEIGHT*2+FIELD_SIZE*FIELD_SIZE)
  offset = 1
  for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
          for z in reverse(1:STACK_REPR_HEIGHT)
              z_idx = z+(FIELD_HEIGHT-STACK_REPR_HEIGHT-1)
              if isnothing(board[x,y,z_idx])
                offset += 2*(z)
                break
              end


              retval[offset:offset+1] = player_to_onehot(stone_player(board[x,y,z_idx]::Tuple{Stone, Player}))        
              offset += 2
          end
          if !isnothing(board[x,y,FIELD_HEIGHT-STACK_REPR_HEIGHT-2])
              retval[offset] = true
          end
          offset += 1
      end
  end
  @assert offset-1 == length(retval)
  retval
end

function onehot_to_stacks!(board::Board, vec::BitArray{1})
  @assert length(vec) == FIELD_SIZE*FIELD_SIZE*STACK_REPR_HEIGHT*2+FIELD_SIZE*FIELD_SIZE
  offset = 1
  for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
          for z in reverse(1:STACK_REPR_HEIGHT)
              player = onehot_to_player(vec[offset:offset+1])
              board[x,y,z+(FIELD_HEIGHT-STACK_REPR_HEIGHT-1)] = isnothing(player) ? nothing : (flat::Stone, player)
              offset += 2
          end
          offset += 1
      end
  end
end

function board_to_enc(board::Board)::BoardEnc
  board = float_board(board)
  topenc = board_top_row_to_onehot(board)
  stacksenc = stacks_to_onehot(board)
  
  vcat(topenc, stacksenc)
end

function enc_to_board(vec::BoardEnc)::Board
  board = empty_board()
  onehot_to_board_top_row!(board, vec[1:6*FIELD_SIZE*FIELD_SIZE])
  onehot_to_stacks!(board, vec[6*FIELD_SIZE*FIELD_SIZE+1:end])
  unfloat_board(board)
end

const board_encoding_length = length(board_to_enc(empty_board()))

const HEIGHTENTROPY = Int(floor(sqrt(FIELD_HEIGHT)))
function compress_board(board::Board)::CompressedBoard
    tmppieceenc = falses(3)
    pieceenc = falses(FIELD_SIZE^2*3)

    heightsum = 0
    notemptysum = 0
    for x in 1:FIELD_SIZE
        for y in 1:FIELD_SIZE
            isnothing(board[x,y,1]) && continue

            height = get_stack_height(board, (x,y))
            topstone = UInt64(board[x,y,height][1])+3*UInt64(board[x,y,height][2])+1

            tmppieceenc.chunks[1] = topstone
            pieceidx = ((x-1)*FIELD_SIZE+(y-1))*3+1
            pieceenc[pieceidx:pieceidx+2] = tmppieceenc

            heightsum += max(height-1, 0)
            notemptysum += height != 0
        end
    end

    tmpheightenc = falses(HEIGHTENTROPY)
    heightenc = falses(notemptysum*HEIGHTENTROPY)
    heightoffset = 1
    stackenc = falses(heightsum+1)
    stackoffset = 1
    for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
        isnothing(board[x,y,1]) && continue
    
        height = 0
        for z in 1:FIELD_HEIGHT
            # We store each stone, at first disregarding whether it was the topstone (which we don't save)
            # If we find out later it was the topstone, reverse the offset back to overwrite the topstone
            s = board[x,y,z]
            if isnothing(s)
                stackoffset -= 1
                break
            end
            s = s::Tuple{Stone, Player}
            stackenc[stackoffset] = Bool(s[2])
            stackoffset += 1
            height += 1
        end
        
        if height == FIELD_HEIGHT
          stackoffset -= 1
        end

        tmpheightenc.chunks[1] = convert(UInt64, height)
        heightenc[heightoffset:heightoffset+HEIGHTENTROPY-1] = tmpheightenc
        heightoffset += HEIGHTENTROPY
      end
    end

    stackenc = stackenc[1:end-1]

    vcat(pieceenc, heightenc, stackenc)
end

function decompress_board(vec::CompressedBoard)::Board
  board = empty_board()
  heights = fill(UInt8(0), FIELD_SIZE, FIELD_SIZE)
  heightoffset = FIELD_SIZE^2*3+1
  for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
          pieceidx = ((x-1)*FIELD_SIZE+(y-1))*3+1
          piece = vec[pieceidx:pieceidx+2].chunks[1]
          
          if piece == 0
              continue
          else
              height = vec[heightoffset:heightoffset+HEIGHTENTROPY-1].chunks[1]
              heightoffset += HEIGHTENTROPY
              heights[x,y] = height
              
              piece -= 1
              board[x,y, height] = (Stone(piece % 3), Player(piece ÷ 3))                
          end
      end
  end
  
  stackoffset = heightoffset
  for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
          if heights[x, y] == 0
              continue
          end
          
          for z in 1:heights[x, y]-1
              board[x,y,z] = (flat::Stone, Player(vec[stackoffset]))
              stackoffset += 1
          end
      end
  end
  board
end


function random_encoded_game(max_moves::Union{Int, Nothing})::Tuple{Union{Result, Nothing}, Array{Tuple{BoardEnc, Array{ActionEnc,1}},1}}
  player = white::Player
  board = empty_board()
  history = Tuple{BoardEnc, Array{ActionEnc,1}}[]

  iter = 0
  while isnothing(max_moves) || iter < max_moves
      possible_actions = action_to_onehot.(enumerate_actions(board, player))

      push!(history, (board_to_enc(board), possible_actions))

      action = rand(possible_actions)
      apply_action!(board, onehot_to_action(action), player)

      result = check_win(board, player)
      if !isnothing(result)
          return result, history
      end
      player = TakEnv.opponent_player(player)
      iter += 1
  end

  nothing, history
end

function random_encoded_game()
  random_encoded_game(nothing)
end


end