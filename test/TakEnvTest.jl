include("../src/TakEnv.jl")

module TakEnvTest
using Test
using ..TakEnv

@testset "TakEnv.jl" begin

  function testboard2()
    board = empty_board()
    board[1,1,1] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[4,3,1] = (TakEnv.flat::Stone, TakEnv.black::Player)
    board[2,1,1] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[4,3,2] = (TakEnv.stand::Stone, TakEnv.black::Player)
    board[3,1,1] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[4,1,1] = (TakEnv.cap::Stone, TakEnv.black::Player)
    board[3,2,1] = (TakEnv.cap::Stone, TakEnv.white::Player)
    board[5,1,1] = (TakEnv.flat::Stone, TakEnv.black::Player)
    board[2,2,1] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[5,1,2] = (TakEnv.flat::Stone, TakEnv.black::Player)
    board[4,2,1] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[4,5,1] = (TakEnv.stand::Stone, TakEnv.black::Player)
    board[5,2,1] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board
  end


  function testboard()
    board = TakEnv.Board(undef, TakEnv.FIELD_SIZE, TakEnv.FIELD_SIZE, TakEnv.FIELD_HEIGHT)

    board[2, 1, 1] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[2, 1, 2] = (TakEnv.flat::Stone, TakEnv.black::Player)
    board[3, 3, 1] = (TakEnv.flat::Stone, TakEnv.black::Player)
    board[4, 1, 1] = (TakEnv.stand::Stone, TakEnv.white::Player)
    board[2, 1, 3] = (TakEnv.cap::Stone, TakEnv.black::Player)
    board[5, 4, 1] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[5, 4, 2] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[5, 4, 3] = (TakEnv.flat::Stone, TakEnv.black::Player)
    board[5, 4, 4] = (TakEnv.flat::Stone, TakEnv.black::Player)
    board[5, 4, 5] = (TakEnv.flat::Stone, TakEnv.white::Player)
    board[5, 4, 6] = (TakEnv.cap::Stone, TakEnv.white::Player)
    board[3, 4, 1] = (TakEnv.stand::Stone, TakEnv.black::Player)
    board[2, 2, 1] = (TakEnv.stand::Stone, TakEnv.white::Player)
    board[5, 3, 1] = (TakEnv.flat::Stone, TakEnv.black::Player)
    board
  end

  @testset "board helpers" begin
    @test TakEnv.get_top_stone(testboard(), (2, 1)) == (TakEnv.cap::Stone, TakEnv.black::Player)
    @test TakEnv.get_top_stone(testboard(), (4, 1)) == (TakEnv.stand::Stone, TakEnv.white::Player)
    @test TakEnv.get_top_stone(testboard(), (3, 1)) === nothing
    @test TakEnv.get_stack_height(testboard(), (3, 1)) == 0
    @test TakEnv.get_stack_height(testboard(), (2, 1)) == 3
    @test TakEnv.stack_height_less_than(testboard(), (2,1), 3) == false
    @test TakEnv.stack_height_less_than(testboard(), (2,1), 4) == true
    @test TakEnv.opponent_player(TakEnv.black::Player) == TakEnv.white::Player
    @test TakEnv.opponent_player(TakEnv.white::Player) == TakEnv.black::Player
    @test TakEnv.board_statistics(TakEnv.rotate_board(testboard(), 2)) == TakEnv.board_statistics(testboard())
  end

  @testset "inverting, rotating, mirroring helpers" begin
    @test TakEnv.rotate_board(testboard(), 3) != testboard()
    @test TakEnv.rotate_board(TakEnv.rotate_board(testboard(), 3), 1) == testboard()
    @test TakEnv.mirror_board(testboard()) != testboard()
    @test TakEnv.mirror_board(TakEnv.mirror_board(testboard())) == testboard()
    @test TakEnv.invert_board(testboard()) != testboard()
    @test TakEnv.invert_board(TakEnv.invert_board(testboard())) == testboard()
    @test TakEnv.invert_board(testboard())[2,1,1] == (TakEnv.flat::Stone, TakEnv.black::Player)
    @test TakEnv.rotate_pos((1, 1)) == (1, TakEnv.FIELD_SIZE)
    @test TakEnv.rotate_pos((TakEnv.FIELD_SIZE, TakEnv.FIELD_SIZE)) == (TakEnv.FIELD_SIZE, 1)
    randpos = (rand(1:TakEnv.FIELD_SIZE), rand(1:TakEnv.FIELD_SIZE))
    @test TakEnv.rotate_pos(TakEnv.rotate_pos(TakEnv.rotate_pos(TakEnv.rotate_pos(randpos)))) == randpos
    if TakEnv.FIELD_SIZE == 5
      @test TakEnv.rotate_pos((3, 3)) == (3, 3)
    end
    @test TakEnv.rotate_action(Action((1, 1), nothing, TakEnv.west::Direction, (1,0,0,0), TakEnv.carry)) == Action((1, 5), nothing, TakEnv.south::Direction, (1,0,0,0), TakEnv.carry)
    @test TakEnv.rotate_action(Action((1, 1), TakEnv.flat::Stone, nothing, nothing, TakEnv.placement::ActionType)) == Action((1, 5), TakEnv.flat::Stone, nothing, nothing, TakEnv.placement::ActionType)
    @test TakEnv.mirror_action(Action((1, 1), nothing, TakEnv.west::Direction, (1,0,0,0), TakEnv.carry)) == Action((5, 1), nothing, TakEnv.east::Direction, (1,0,0,0), TakEnv.carry)    
    @test TakEnv.mirror_action(Action((1, 1), nothing, TakEnv.south::Direction, (1,0,0,0), TakEnv.carry)) == Action((5, 1), nothing, TakEnv.south::Direction, (1,0,0,0), TakEnv.carry)
    @test TakEnv.mirror_action(Action((1, 1), TakEnv.flat::Stone, nothing, nothing, TakEnv.placement::ActionType)) == Action((5, 1), TakEnv.flat::Stone, nothing, nothing, TakEnv.placement::ActionType)

  end

  @testset "board_statistics" begin
    @test TakEnv.board_statistics(testboard(), TakEnv.white::Player) == Dict(
          TakEnv.flat::Stone => 4,
          TakEnv.stand::Stone => 2,
          TakEnv.cap::Stone => 1
      )
    @test TakEnv.board_statistics(testboard(), TakEnv.black::Player) == Dict(
          TakEnv.flat::Stone => 5,
          TakEnv.stand::Stone => 1,
          TakEnv.cap::Stone => 1
      )

    @test TakEnv.board_statistics(testboard(), nothing) == Dict(
          TakEnv.flat::Stone => 9,
          TakEnv.stand::Stone => 3,
          TakEnv.cap::Stone => 2
      )
  end

  @testset "check carry helpers" begin
    @test TakEnv.check_outside_board((1, 0)) == true
    @test TakEnv.check_outside_board((TakEnv.FIELD_SIZE+1, TakEnv.FIELD_SIZE)) == true
    @test TakEnv.check_outside_board((4, 4)) == false

    @test TakEnv.check_movement(testboard(), (3, 3), TakEnv.south::Direction) == false
    @test TakEnv.check_movement(testboard(), (3, 3), TakEnv.east::Direction) == true
    @test TakEnv.check_movement(testboard(), (2, 1), TakEnv.north::Direction) == false
    @test TakEnv.check_movement(testboard(), (2, 1), TakEnv.south::Direction) == true

    @test TakEnv.carry_only_zeros((0,0,0)) == true
    @test TakEnv.carry_only_zeros(()) == true
  end


  @testset "enumerate_actions" begin
    @test length(enumerate_actions(Board(undef, TakEnv.FIELD_SIZE, TakEnv.FIELD_SIZE, TakEnv.FIELD_HEIGHT), TakEnv.white::Player)) == TakEnv.FIELD_SIZE ^ 2 * 3

    @test begin
      board = testboard()
      issubset(enumerate_actions(board, TakEnv.white::Player), enumerate_actions(board)) &&
      issubset(enumerate_actions(board, TakEnv.black::Player), enumerate_actions(board))
    end

    @test unique(enumerate_actions(testboard())) == enumerate_actions(testboard())
    @test unique(enumerate_actions(testboard(), TakEnv.white::Player)) == enumerate_actions(testboard(), TakEnv.white::Player)

    # These tests only work for a 5x5 board
    if TakEnv.FIELD_SIZE == 5
      @testset "check_carry" begin
        @test Action((2, 1), nothing, TakEnv.west::Direction, (1,0,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player)
        @test !(Action((2, 1), nothing, TakEnv.west::Direction, (1,1,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player))
        @test (Action((2, 1), nothing, TakEnv.south::Direction, (1,0,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player))
        @test !(Action((2, 1), nothing, TakEnv.south::Direction, (1,1,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player))
        @test !(Action((2, 1), nothing, TakEnv.south::Direction, (2,0,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player))
        @test (Action((2, 1), nothing, TakEnv.east::Direction, (2,1,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player))
        @test !(Action((2, 1), nothing, TakEnv.east::Direction, (1,1,1,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player))
        @test (Action((2, 1), nothing, TakEnv.east::Direction, (3,0,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player))
        @test !(Action((2, 1), nothing, TakEnv.east::Direction, (4,0,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.black::Player))
        @test !(Action((2, 2), nothing, TakEnv.north::Direction, (1,0,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.white::Player))
        @test (Action((2, 2), nothing, TakEnv.east::Direction, (1,0,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.white::Player))
        @test (Action((5, 4), nothing, TakEnv.north::Direction, (2,2,1,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.white::Player))
        @test (Action((5, 4), nothing, TakEnv.west::Direction, (-1,0,0,0), TakEnv.carry) in TakEnv.enumerate_actions(testboard(), TakEnv.white::Player))
      end
    end
  end

  # These tests only work on a 5x5 board
  if TakEnv.FIELD_SIZE ==  5
    @testset "apply action" begin
      @test TakEnv.board_statistics(apply_action!(testboard(), Action((1,1), TakEnv.flat::Stone, nothing, nothing, TakEnv.placement::ActionType), TakEnv.white::Player), TakEnv.white::Player) == Dict(TakEnv.flat::Stone => 5, TakEnv.cap::Stone => 1, TakEnv.stand::Stone => 2)
      @test TakEnv.get_stack_height(apply_action!(testboard(), Action((5, 4), nothing, TakEnv.west::Direction, (-1,0,0,0), TakEnv.carry::ActionType), TakEnv.white::Player), (4,4)) == 6
      @test TakEnv.get_stack_height(apply_action!(testboard(), Action((5, 4), nothing, TakEnv.west::Direction, (-1,0,0,0), TakEnv.carry::ActionType), TakEnv.white::Player), (5,4)) == 0
      @test TakEnv.get_stack_height(apply_action!(testboard(), Action((2, 1), nothing, TakEnv.west::Direction, (2,0,0,0), TakEnv.carry::ActionType), TakEnv.black::Player), (1,1)) == 2
      @test TakEnv.get_stack_height(apply_action!(testboard(), Action((2, 1), nothing, TakEnv.west::Direction, (2,0,0,0), TakEnv.carry::ActionType), TakEnv.black::Player), (2,1)) == 1
      @test TakEnv.stone_type(apply_action!(testboard(), Action((2,1), nothing, TakEnv.south::Direction, (1,0,0,0), TakEnv.carry::ActionType), TakEnv.black::Player)[2, 2, 1]) == TakEnv.flat::Stone
    end
  end

  # These tests only work on a 5x5 board
  if TakEnv.FIELD_SIZE == 5
    @testset "road_win helpers" begin
      @test TakEnv.check_road_search(testboard2(), (1,1), NTuple{2, Int}[], true, TakEnv.white::Player) == true
      @test TakEnv.check_road_search(testboard2(), (4,1), NTuple{2, Int}[], false, TakEnv.black::Player) == false
      @test TakEnv.check_road_win(testboard2(), TakEnv.white::Player) == true
      @test TakEnv.check_road_win(testboard2(), TakEnv.black::Player) == false
      @test TakEnv.check_road_win(testboard(), TakEnv.white::Player) == false
      @test TakEnv.check_road_win(testboard(), TakEnv.black::Player) == false
    end
  end

  @testset "flat_win helpers" begin
    @test TakEnv.check_fully_covered(testboard()) == false
    @test TakEnv.check_player_out_of_stones(testboard()) == false
    @test TakEnv.count_flats(testboard()) == Dict(TakEnv.white::Player => 0, TakEnv.black::Player => 2+TakEnv.KOMI)
  end

  # These tests only work on 5x5 board
  if TakEnv.FIELD_SIZE == 5
    @testset "check_win" begin
      @test TakEnv.check_win(testboard(), TakEnv.white::Player) === nothing
      @test TakEnv.check_win(testboard2(), TakEnv.white::Player) == (TakEnv.road_win::ResultType, TakEnv.white::Player) 
    end 
  end

  @testset "check_stalemate" begin
    @test TakEnv.check_stalemate([testboard() for _ in 1:10]) == true
    @test TakEnv.check_stalemate([testboard() for _ in 1:3]) == false
  end

  @testset "random_game" begin
    @test length(TakEnv.random_game(4)[3]) == 4
    @test length(TakEnv.random_game()[3]) >= TakEnv.FIELD_SIZE*2-1
    @test !isnothing(TakEnv.random_game()[2])
    @test length(TakEnv.random_game(40)[3]) <= 40
  end

  @testset "render" begin
    @test typeof(TakEnv.render_board(random_game()[1])) == TakEnv.Luxor.Drawing
  end
end
end