include("../src/TakInterface.jl")

module TakInterfaceTest

using ..TakInterface
using ..TakInterface: Board, Action, get_init_board, get_board_size, get_action_size, get_next_state, get_valid_moves, get_game_ended, get_canonical_form, get_symmetries, get_compressed_representation
using ..Encoder

using Test

@testset "TakInterface.jl" begin
  @testset "play an example game" begin
    board = get_init_board()
    player = 1
    # The encoding length should match
    @test size(board)[1] == Encoder.board_encoding_length

    # There should be valid moves
    @test any(get_valid_moves(board, player))
    @test get_game_ended(board, player) == 0

    moves = get_valid_moves(board, player)

    # Action size should match
    @test size(moves)[1] == get_action_size()

    # Pick a random move
    current_move = falses(get_action_size())
    current_move[rand(findall(moves))] = 1

    # Something should have changed when transitioning state
    @test get_next_state(board, player, current_move) != board
    board = get_next_state(board, player, current_move)
    player = -player

    # The last move should not be available anymore
    @test (current_move .& get_valid_moves(board, player)) == falses(get_action_size())

    # The canonical state should be different from the view of another player
    @test get_canonical_form(board, player) != get_canonical_form(board, -player)

    # Pick another move, now the board must be asymmetric
    current_move = falses(get_action_size())
    current_move[findfirst(get_valid_moves(board, player))] = 1
    board = get_next_state(board, player, current_move)
    player = -player


    # Test symmetries
    pi = get_valid_moves(board, player)
    @test size(get_symmetries(board, pi))[1] == 8
    @test get_symmetries(board, pi) == unique(get_symmetries(board, pi))

    @test size(get_compressed_representation(board)) < size(board)


  end



end


end