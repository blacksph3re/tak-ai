module TakUI

using Crayons
using Luxor
using ..TakEnv
using ..TakEnv: FIELD_SIZE, FIELD_HEIGHT, Player, white, black, Stone, flat, stand, cap, Board
using ..Encoder

# Renders a board
const RENDER_RESOLUTION = 1000
const RENDER_SIZE = Int(floor(RENDER_RESOLUTION/(FIELD_SIZE+1)))
const OFFSET_3D = 10
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


# Turns the Tak board into a string
const CMD_CELL_WIDTH = 5
function player_color(p::TakEnv.Player)
   p == white::Player ? crayon"light_blue" : crayon"light_red"
end

function board_color()
  crayon"white"
end

function render_board_cmd(board::Board)::String
  io = IOBuffer()
  max_height = maximum([TakEnv.get_stack_height(board, (x, y)) for x in 1:FIELD_SIZE for y in 1:FIELD_SIZE])
  stackrows = convert(Int, ceil((max_height-1) / CMD_CELL_WIDTH))
  row_width = CMD_CELL_WIDTH * FIELD_SIZE + FIELD_SIZE

  board = Encoder.float_board(board)


  for y in 1:FIELD_SIZE
    # Render divider
    if y != 1
      print(io, board_color(),join(["-" for _ in 1:row_width]))
      print(io, "\n")
    end

    # Render the topmost stone
    for x in 1:FIELD_SIZE
      if x != 1
        print(io, board_color(), "|")
      end
      if isnothing(board[x,y,FIELD_HEIGHT])
        print(io, join([" " for _ in 1:CMD_CELL_WIDTH]))
      else
        print(io, player_color(TakEnv.stone_player(board[x, y, FIELD_HEIGHT])), rpad(string(TakEnv.stone_type(board[x, y, FIELD_HEIGHT])), CMD_CELL_WIDTH))
      end
    end
    print(io, "\n")

    for stackrow in 1:stackrows
      for x in 1:FIELD_SIZE
        if x != 1
          print(io, board_color(), "|")
        end

        startidx = (stackrow - 1) * CMD_CELL_WIDTH + 1
        for i in 1:CMD_CELL_WIDTH
          if startidx + i >= FIELD_HEIGHT
            print(io, " ")
          else
            stone = board[x, y, FIELD_HEIGHT-startidx-i]
            if isnothing(stone)
              print(io, " ")
            else
              print(io, player_color(TakEnv.stone_player(stone)), "*")
            end
          end
        end
      end
      print(io, "\n")
    end
  end

  String(take!(io))
end

# Turns an action to a human readable format
function action_string(action::TakEnv.Action)::String
  if action.action_type == TakEnv.placement::ActionType
    "place $(string(action.stone)) $(action.pos)"
  else
    if action.carry[1] == -1
      "move $(action.pos) $(string(action.dir))"
    else
      "move $(action.pos) $(string(action.dir)) drop $(join([string(c) for c in action.carry if c != 0], " "))"
    end
  end
end

# Parses an action from a string
function read_action(input::String)::TakEnv.Action
  placement_regex = r"^\s?place\s(?'stone_type'flat|stand|cap)\s\((?'x'\d),\s?(?'y'\d)\)\s*$"
  carry_all_regex = r"^\s?move\s\((?'x'\d),\s?(?'y'\d)\)\s(?'dir'north|east|south|west)\s*$"
  carry_regex = r"^\s?move\s\((?'x'\d),\s?(?'y'\d)\)\s(?'dir'north|east|south|west)\sdrop(?'drop'(\s\d)+)\s*$"

  m1 = match(placement_regex, input)
  m2 = match(carry_all_regex, input)
  m3 = match(carry_regex, input)

  action = if !isnothing(m1)
    # It's a placement
    type = [x for x in instances(TakEnv.Stone) if string(x) == m1["stone_type"]][1]
    pos = (parse(Int, m1["x"]), parse(Int, m1["y"]))

    TakEnv.Action(pos, type, nothing, nothing, TakEnv.placement)
  elseif !isnothing(m2)
    # Carry the entire stack
    carry = zeros(FIELD_SIZE-1)
    carry[1] = -1
    carry = tuple(carry...)

    dir = [x for x in instances(TakEnv.Direction) if string(x) == m2["dir"]][1]

    pos = (parse(Int, m2["x"]), parse(Int, m2["y"]))

    TakEnv.Action(pos, nothing, dir, carry, TakEnv.carry)
  elseif !isnothing(m3)
    carry_input = m3["drop"]
    carry_input = replace(carry_input, r"\s" => "")
    @assert length(carry_input) <= FIELD_SIZE-1 "too many drops, maximum is $(FIELD_SIZE-1)"

    carry = zeros(FIELD_SIZE-1)
    for i in 1:min(length(carry_input), FIELD_SIZE-1)
      carry[i] = parse(Int, carry_input[i])
    end
    carry = tuple(carry...)

    dir = [x for x in instances(TakEnv.Direction) if string(x) == m3["dir"]][1]

    pos = (parse(Int, m3["x"]), parse(Int, m3["y"]))

    TakEnv.Action(pos, nothing, dir, carry, TakEnv.carry)
  else
    @assert false "action not matching any format, use either 'place <stone> <pos>', 'move <pos> <dir>' or 'move <pos> <dir> drop <carries>'"
  end


  TakEnv.assert_action(action)
  action
end



function perform_move(state::Tuple{Board, Player}, move::String)::Tuple{Board, Player}
  action = read_action(move)
  board = copy(state[1])
  TakEnv.apply_action!(board, action, state[2])

  print("Current player $(state[2] == TakEnv.white ? "white" : "black")\n")
  print(render_board_cmd(board))

  (board, TakEnv.opponent_player(state[2]))
end

function enumerate_actions(state::Tuple{Board, Player})
  actions = action_string.(TakEnv.enumerate_actions(state[1], state[2]))
  print(actions)
end

end