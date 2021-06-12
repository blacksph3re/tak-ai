

module TakEnv
using Luxor

export Action, Result, Board, CarryType
export empty_board, enumerate_actions, apply_action!, check_win, check_stalemate, render_board, random_game, opponent_player
export Stone, Player, Direction, ActionType, ResultType
export white, black

# const FIELD_SIZE = 8
# const NORMAL_PIECE_COUNT = 50
# const CAPSTONE_COUNT = 2

# const FIELD_SIZE = 7
# const NORMAL_PIECE_COUNT = 40
# const CAPSTONE_COUNT = 2

# const FIELD_SIZE=6
# const NORMAL_PIECE_COUNT=30
# const CAPSTONE_COUNT = 1

const FIELD_SIZE = 5
const NORMAL_PIECE_COUNT = 21
const CAPSTONE_COUNT = 1

# A score which is added to black when evaluating a win to compensate first-player-advantage
const KOMI = 2

# The highest possible tower is all normal pieces stacked as flat stones with a capstone on it
const FIELD_HEIGHT = NORMAL_PIECE_COUNT*2+1


@enum Stone::UInt8 flat stand cap
@enum Player::UInt8 white black
@enum Direction::UInt8 north east south west
@enum ActionType::UInt8 placement carry
@enum ResultType::UInt8 road_win flat_win double_road_win draw

CarryType = NTuple{FIELD_SIZE-1, Int8}
Result = Tuple{ResultType, Union{Player, Nothing}}

struct Action
    pos::NTuple{2, Int}
    stone::Union{Stone, Nothing}
    dir::Union{Direction, Nothing}
    carry::Union{CarryType, Nothing}
    action_type::ActionType
end

Board = Array{Union{Tuple{Stone, Player}, Nothing}, 3}

function calc_possible_carries()::Array{CarryType,1}
  retval = CarryType[]
  push!(retval, Tuple(vcat([-1], zeros(FIELD_SIZE-2))))
  for carry in Iterators.product([0:FIELD_SIZE for _ in 1:(FIELD_SIZE-1)]...)
      if sum(carry) > FIELD_SIZE || sum(carry)==0
          continue
      end
      
      zero_error = false
      last_zero = false
      for c in carry
          if last_zero && c != 0
              zero_error = true
              break
          end
          if c == 0
              last_zero = true
          end
      end
      
      if zero_error
          continue
      end
      
      push!(retval, carry)
  end
  
  retval
end
const possible_carries = calc_possible_carries()
const possible_directions = [north::Direction, east::Direction, south::Direction, west::Direction]

function empty_board()
    Board(undef, FIELD_SIZE, FIELD_SIZE, FIELD_HEIGHT)
end


function get_top_stone(board::Board, pos::NTuple{2, Int})::Union{Tuple{Stone, Player}, Nothing}
  for z in 2:FIELD_HEIGHT
      if isnothing(board[pos[1], pos[2], z])
          return board[pos[1], pos[2], z-1]
      end
  end
  return board[pos[1], pos[2], FIELD_HEIGHT]
end

function get_top_stone(board::Board, pos::NTuple{2, Int}, stack_height_greater::Int)::Tuple{Stone, Player}
    for z in stack_height_greater+1:FIELD_HEIGHT
        if isnothing(board[pos[1], pos[2], z])
            return board[pos[1], pos[2], z-1]
        end
    end
    return board[pos[1], pos[2], FIELD_HEIGHT]
end

function stone_type(stone::Union{Tuple{Stone, Player}, Nothing})::Union{Stone, Nothing}
  if isnothing(stone)
      return nothing
  end
  stone[1]
end

function stone_player(stone::Union{Tuple{Stone, Player}, Nothing})::Union{Player, Nothing}
  if isnothing(stone)
      return nothing
  end
  stone[2]
end

function get_stack_height(board::Board, pos::NTuple{2, Int})::Int
  retval = 0
  for z in 1:FIELD_HEIGHT
      if isnothing(board[pos[1], pos[2], z])
          break
      end
      retval += 1
  end
  retval
end

function stack_height_less_than(board::Board, pos::NTuple{2, Int}, max::Int)::Bool
    for z in 1:max
        if isnothing(board[pos[1], pos[2], z])
            return true
        end
    end
    return false
end

function opponent_player(player::Player)::Player
  Player((Int(player)+1)%2)
end

# Rotate the entire board 90 degrees ccw
function rotate_board(board::Board, rotations::Int)::Board
    board2 = copy(board)
    for i in 1:TakEnv.FIELD_HEIGHT
        board2[:,:,i] = rotl90(board[:,:,i], rotations)
    end
    board2
end

# Rotate a direction 90 degrees ccw
function rotate_direction(dir::Direction)::Direction
    Direction((Int(dir)+3)%4)
end
rotate_direction(nothing) = nothing

# Rotate a position 90 degrees ccw
# Use the fact that rotation around (0, 0) by 90 degrees works through
# changing A(x, y) to A'(-y, x) -> Reshape this equation to fit our coordinate
# system
function rotate_pos(pos::NTuple{2, Int})::NTuple{2, Int}
    return (pos[2], -(pos[1]-1) + FIELD_SIZE)
end

# Rotate an action 90 degrees ccw
function rotate_action(action::Action)::Action
    Action(
        rotate_pos(action.pos),
        action.stone,
        rotate_direction(action.dir),
        action.carry,
        action.action_type)
end

# Mirror a board left to right
function mirror_board(board::Board)::Board
    board[end:-1:1, :, :]
end

# Mirror a direction left to right
function mirror_direction(dir::Direction)::Direction
    if dir == north::Direction || dir == south::Direction
        dir
    else
        rotate_direction(rotate_direction(dir))
    end
end
mirror_direction(nothing) = nothing


# Mirror a position left to right
function mirror_pos(pos::NTuple{2, Int})::NTuple{2, Int}
    (-(pos[1]-1) + FIELD_SIZE, pos[2])
end

# Mirror an action left to right
function mirror_action(action::Action)::Action
    Action(
        mirror_pos(action.pos),
        action.stone,
        mirror_direction(action.dir),
        action.carry,
        action.action_type
    )
end

function invert_stone(stone::Union{Tuple{Stone, Player}, Nothing})::Union{Tuple{Stone, Player}, Nothing}
    if isnothing(stone)
        return nothing
    end
    return (stone[1], opponent_player(stone[2]))
end

function invert_board(board::Board)::Board
    invert_stone.(board)
end

# Renders a board
const RENDER_RESOLUTION = 1000
const RENDER_SIZE = Int(floor(RENDER_RESOLUTION/(FIELD_SIZE+1)))
function render_board(board::Board)
    @draw begin
        background("grey15")
        origin(Point(0, 0))
        setline(1)
        for x in range(1, stop=FIELD_SIZE)
            for y in range(1, stop=FIELD_SIZE)
                #sethue((x+FIELD_SIZE*y) % 2 == 0 ? "red" : "blue")
                sethue("chocolate1")
                setopacity(1)
                origin(Point(x * RENDER_SIZE, y * RENDER_SIZE))
                setline(1)
                p = hypotrochoid(20, 5, 11, :stroke, stepby=0.01, vertices=true)
                for i in 0:2:15
                    poly(offsetpoly(p, i), :stroke, close=true)
                end

                sethue("blanchedalmond")
                setopacity(0.5)
                setline(4)
                if(y < FIELD_SIZE)
                    line(Point(0, RENDER_SIZE/5),
                         Point(0, RENDER_SIZE - RENDER_SIZE/5), :stroke)
                end
                if(x < FIELD_SIZE)
                    line(Point(RENDER_SIZE/5, 0),
                         Point(RENDER_SIZE - RENDER_SIZE/5, 0), :stroke)
                end
            end
        end

        origin(Point(0, 0))
        setopacity(1)

        OFFSET_3D = 10

        for z in range(1, stop=FIELD_HEIGHT)
            for x in range(1, stop=FIELD_SIZE)
                for y in range(1, stop=FIELD_SIZE)
                    if !isnothing(board[x, y, z])
                        stone, player = board[x, y, z]
                        sethue(player == white::Player ? "cornsilk" : "honeydew4")
                        origin(Point(x*RENDER_SIZE + (z-1)*OFFSET_3D, y*RENDER_SIZE - (z-1)*OFFSET_3D))
                        if stone == flat::Stone
                            box(Point(0, 0), RENDER_SIZE/2, RENDER_SIZE/2, 8, :fill)
                            sethue("grey15")
                            setline(0.5)
                            box(Point(0, 0), RENDER_SIZE/2, RENDER_SIZE/2, 8, :stroke)
                        elseif stone == stand::Stone
                            rotate(π/4)
                            box(Point(0, 0), RENDER_SIZE/5, RENDER_SIZE/2, 3, :fill)
                            sethue("grey15")
                            setline(0.5)
                            box(Point(0, 0), RENDER_SIZE/5, RENDER_SIZE/2, 3, :stroke)
                        elseif stone == cap::Stone
                            box(Point(0, 0), RENDER_SIZE/2, RENDER_SIZE/2, 8, :fill)
                            sethue("grey15")
                            setline(0.5)
                            box(Point(0, 0), RENDER_SIZE/2, RENDER_SIZE/2, 8, :stroke)
                            sethue("gold")
                            circle(Point(0, 0), RENDER_SIZE/8, :fill)
                        end
                    end
                end
            end
        end
    end 1000 1000
end

function assert_action(a::Action)
  @assert a.dir != north::Direction || a.pos.y != 1 "can't move north with y=1"
  @assert a.dir != south::Direction || a.pos.y != FIELD_SIZE "can't move south with y=FIELD_SIZE"
  @assert a.dir != east::Direction || a.pos.x != FIELD_SIZE "can't move east with x=FIELD_SIZE"
  @assert a.dir != west::Direction || a.pos.x != 1 "can't move west with x=1"
  
  @assert a.pos.x <= FIELD_SIZE && a.pos.x >= 1 "invalid x position"
  @assert a.pos.y <= FIELD_SIZE && a.pos.y >= 1 "invalid y position"
  
  @assert a.action_type != placement::ActionType || !isnothing(a.stone) "a placement needs a stone"
  @assert a.action_type != movement::ActionType || !isnothing(a.dir) "a movement needs a direction"
  @assert a.action_type != carry::ActionType || (!isnothing(a.dir) && !isnothing(a.carry)) "a carry needs a carry and a dir"
  
  @assert !(isnothing(a.stone) && isnothing(a.dir) && isnothing(a.carry)) "At least one action type needs data"
  @assert !(!isnothing(a.stone) && (!isnothing(a.dir) || !isnothing(a.carry))) "If it's a placement, it can't have a move or carry"
  @assert !(!isnothing(a.carry) && !isnothing(a.dir)) "If it's a carry, it needs a direction"
  
  @assert isnothing(a.carry) || !isnothing(findfirst(c -> c == a.carry, possible_carries)) "Impossible carry"
  true
end


# Calculate some board statistics
function board_statistics(board::Board, player::Union{Player, Nothing})::Dict{Stone, Int}
    spent_stones = fill(0::Int, length(instances(Stone)))

    for x in 1:FIELD_SIZE
        for y in 1:FIELD_SIZE
            for z in 1:FIELD_HEIGHT
                stone = board[x,y,z]
                if isnothing(stone)
                    break
                end
                spent_stones[Int(stone_type(stone))+1] += (isnothing(player) || stone_player(stone) == player)
            end
        end
    end

    Dict(
        flat::Stone => spent_stones[Int(flat::Stone)+1],
        stand::Stone => spent_stones[Int(stand::Stone)+1],
        cap::Stone => spent_stones[Int(cap::Stone)+1],
    )
end
function board_statistics(board::Board)::Dict{Stone, Int}
    board_statistics(board, nothing)
end


function translate_pos(pos::NTuple{2, Int}, dir::Direction)::NTuple{2, Int}
    if dir==north::Direction
        (pos[1], pos[2]-1)
    elseif dir==east::Direction
        (pos[1]+1, pos[2])
    elseif dir==south::Direction
        (pos[1], pos[2]+1)
    elseif dir==west::Direction
        (pos[1]-1, pos[2])
    end
end

function check_outside_board(pos::NTuple{2, Int})::Bool
    pos[1] < 1 || pos[2] < 1 || pos[1] > FIELD_SIZE || pos[2] > FIELD_SIZE
end


function check_movement(board::Board, pos::NTuple{2, Int}, dir::Direction)::Bool
    newpos = translate_pos(pos, dir)
    !check_outside_board(newpos) && 
    stone_type(get_top_stone(board, newpos)) != cap::Stone && 
    (stone_type(get_top_stone(board, newpos)) != stand::Stone || stone_type(get_top_stone(board, pos)) == cap::Stone)
end



function carry_only_zeros(carry::Tuple)::Bool
  for x in carry
      if x != 0
          return false
      end
  end
  true
end



function check_carry_rec(board::Board, pos::NTuple{2, Int}, dir::Direction, carry::Tuple, topstone::Stone)::Bool
  # Recursion end
  if carry==() || carry_only_zeros(carry)
      return true
  end
  
  curpos = translate_pos(pos, dir)
  
  # End of the field
  if check_outside_board(curpos)
      return false
  end
  
  # Capstone
  if stone_type(get_top_stone(board, curpos)) == cap::Stone
      return false
  end
  
  # A stand stone can be flattened by a capstone if it is the only one
  if stone_type(get_top_stone(board, curpos)) == stand::Stone
      return topstone == cap::Stone && carry[1] == 1 && carry_only_zeros(carry[2:end])
  end
  
  # Otherwise, check recursively
  check_carry_rec(board, curpos, dir, carry[2:end], topstone)
end


function check_carry(board::Board, pos::NTuple{2, Int}, dir::Direction, carry::CarryType, topstone::Stone)::Bool
    if carry[1] == -1
        return get_stack_height(board, pos) > FIELD_SIZE && check_carry_rec(board, pos, dir, carry, stone_type(get_top_stone(board, pos)))
    end

    check_carry_rec(board, pos, dir, carry, topstone)
end



function enumerate_actions(board::Board, player::Player)::Array{Action,1}
    retval = Action[]
    sizehint!(retval, 50)
    
    stats = board_statistics(board, player)

    # Enumerate all possible carries
    for x in 1:FIELD_SIZE
        for y in 1:FIELD_SIZE
            if isnothing(board[x, y, 1])
                if stats[flat::Stone] + stats[stand::Stone] < NORMAL_PIECE_COUNT
                    push!(retval, Action((x,y), flat::Stone, nothing, nothing, placement::ActionType))
                    push!(retval, Action((x,y), stand::Stone, nothing, nothing, placement::ActionType))
                end
                if stats[cap::Stone] < CAPSTONE_COUNT
                    push!(retval, Action((x,y), cap::Stone, nothing, nothing, placement::ActionType))
                end
            else
                top_stone = get_top_stone(board, (x, y))
                if stone_player(top_stone) == player
                    stack_height = get_stack_height(board, (x,y))
                    for c in possible_carries
                        carry_height = convert(Int, sum(c))
                        if stack_height < carry_height
                            continue
                        end
                        for dir in possible_directions
                            if check_carry(board, (x, y), dir, c, stone_type(top_stone))
                                push!(retval, Action((x, y), nothing, dir, c, carry::ActionType))
                            end
                        end
                    end
                end
            end
        end
    end
  
    retval
end

function enumerate_actions(board::Board)::Vector{Action}
  union(enumerate_actions(board, white::Player), enumerate_actions(board, black::Player))
end


# Apply an acction
# Does not check for validity of that action, do so with enumerate_actions
function apply_action!(board::Board, action::Action, player::Player)
  statistics = board_statistics(board, player)
  all_statistics = board_statistics(board, nothing)
  
  total_stones = sum([i[2][2] for i in enumerate(all_statistics)])
  
  if action.action_type == placement::ActionType
    @assert isnothing(board[action.pos[1], action.pos[2], 1])
    if total_stones < 2 # The first two rounds you place the opponents stone
        board[action.pos[1], action.pos[2], 1] = (action.stone, opponent_player(player))
    else
        board[action.pos[1], action.pos[2], 1] = (action.stone, player)
    end
  else
    # Treat the (-1,0,0,0) carry separately
    if action.carry[1] == -1
        newpos = translate_pos(action.pos, action.dir)
        for z in 1:FIELD_HEIGHT
            board[newpos[1], newpos[2], z] = board[action.pos[1], action.pos[2], z]
            board[action.pos[1], action.pos[2], z] = nothing
        end
    else
        # First pick up stones from the origin stack
        picked_up_stack = []
        for z in reverse(1:FIELD_HEIGHT)
            if !isnothing(board[action.pos[1], action.pos[2], z])
                push!(picked_up_stack, board[action.pos[1], action.pos[2], z])
                board[action.pos[1], action.pos[2], z] = nothing
            end
            if length(picked_up_stack) >= sum(action.carry)
                break
            end
        end
        
        @assert(length(picked_up_stack) == sum(action.carry))
        
        # Then loop through the carry and drop according to the carry
        curpos = action.pos
        for c in action.carry
            curpos = translate_pos(curpos, action.dir)
            if c == 0
                continue
            end
            
            top = get_stack_height(board, curpos)
            
            if top >= 1 && stone_type(board[curpos[1], curpos[2], top]) == stand::Stone
                @assert stone_type(last(picked_up_stack)) == cap::Stone
                board[curpos[1], curpos[2], top] = (flat::Stone, stone_player(board[curpos[1], curpos[2], top]))
            end
            
            for i in 1:c
                board[curpos[1], curpos[2], top+i] = pop!(picked_up_stack)
            end
        end
    end
  end
      
  board
end


function check_road_search(board::Board, pos::NTuple{2, Int}, already_checked::Array{NTuple{2, Int}}, horizontal::Bool, player::Player)::Bool
  stone = get_top_stone(board, pos)
  # Surely not a road
  if isnothing(stone) || stone_type(stone) == stand::Stone || stone_player(stone) != player
      return false
  end
  # Surely already a road
  if (horizontal && pos[1] == FIELD_SIZE) || (!horizontal && pos[2] == FIELD_SIZE)
      return true
  end
  to_check = [p for p in [translate_pos(pos, dir) for dir in possible_directions] if !check_outside_board(p) && !(p in already_checked)]

  any(map(p -> check_road_search(board, p, vcat(to_check, already_checked), horizontal, player), to_check))
end

# Check for a road-win situation
function check_road_win(board::Board, player::Player)::Bool
  # Start with all left and top stones and search for a road from there
  for coord in 1:FIELD_SIZE
      # vertical road
      top = get_top_stone(board, (coord, 1))
      if !isnothing(top) && stone_player(top) == player && check_road_search(board, (coord, 1), NTuple{2,Int}[], false, player)
          return true
      end
      
      # horizontal road
      top = get_top_stone(board, (1, coord))
      if !isnothing(top) && stone_player(top) == player && check_road_search(board, (1, coord), NTuple{2,Int}[], true, player)
          return true
      end
  end
  false
end

function check_fully_covered(board::Board)::Bool
  for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
          if isnothing(board[x, y, 1])
              return false
          end
      end
  end
  true
end

function check_player_out_of_stones(board::Board)::Bool
  stats_white = board_statistics(board, white::Player)
  stats_black = board_statistics(board, black::Player)
  
  (stats_white[cap::Stone] >= CAPSTONE_COUNT &&
  stats_white[flat::Stone] + stats_white[stand::Stone] >= NORMAL_PIECE_COUNT) ||
  (stats_black[cap::Stone] >= CAPSTONE_COUNT &&
  stats_black[flat::Stone] + stats_black[stand::Stone] >= NORMAL_PIECE_COUNT)
end
  
function count_flats(board::Board)::Dict{Player, Int}
  retval = Dict(white::Player => 0, black::Player => KOMI)
  for x in 1:FIELD_SIZE
      for y in 1:FIELD_SIZE
          top = get_top_stone(board, (x,y))
          if stone_type(top) == flat::Stone
              retval[stone_player(top)] += 1
          end
      end
  end
  retval
end

function check_win(board::Board, active_player::Player)::Union{Result, Nothing}
  road_win_white = check_road_win(board, white::Player)
  road_win_black = check_road_win(board, black::Player)
  
  if road_win_white && road_win_black
      return (double_road_win::ResultType, active_player)
  end
  
  if road_win_white
      return (road_win::ResultType, white::Player)
  end
  
  if road_win_black
      return (road_win::ResultType, black::Player)
  end
  
  if check_fully_covered(board) || check_player_out_of_stones(board)
      flats = count_flats(board)
      if flats[white::Player] > flats[black::Player]
          return (flat_win::ResultType, white::Player)
      end
                  
      if flats[black::Player] > flats[white::Player]
          return (flat_win::ResultType, black::Player)
      end
      
      return (draw::ResultType, nothing)
  end
  
  nothing
end

function check_stalemate(state_history)::Bool
    uniques = unique(state_history)
    any(x -> x > 5, (count(s -> u==s, state_history) for u in uniques))
end

function random_game(max_moves::Union{Int, Nothing})::Tuple{Board, Union{Result, Nothing}, Array{Action,1}}
    player = white::Player
    board = empty_board()
    actions_taken = Action[]

    iter = 0
    while isnothing(max_moves) || iter < max_moves
        possible_actions = enumerate_actions(board, player)
        action = rand(possible_actions)
        push!(actions_taken, action)

        apply_action!(board, action, player)

        result = check_win(board, player)
        if !isnothing(result)
            return board, result, actions_taken
        end
        player = opponent_player(player)
        iter += 1
    end

    board, nothing, actions_taken
end

function random_game()::Tuple{Board, Union{Result, Nothing}, Array{Action,1}}
    random_game(nothing)
end

end