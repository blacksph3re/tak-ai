
module EncoderTest

using ..Encoder
using ..Encoder: onehot_to_pos, pos_to_onehot, idx_to_pos, pos_to_idx, stone_to_onehot, onehot_to_stone, direction_to_onehot, onehot_to_direction, carry_to_onehot, onehot_to_carry
using ..Encoder: float_board, unfloat_board, piece_to_onehot, onehot_to_piece, board_top_row_to_onehot, onehot_to_board_top_row!, STACK_REPR_HEIGHT, player_to_onehot, onehot_to_player, stacks_to_onehot, onehot_to_stacks!, random_encoded_game
using ..TakEnv
using ..TakEnv: FIELD_SIZE, FIELD_HEIGHT, possible_carries, possible_directions, stone_player, stone_type, flat, cap, stand, north, south, east, west, placement, carry, white, black

using Test

@testset "Encoder.jl" begin
  function testboard()
    board = empty_board()
    board[1,1,1] = (flat::Stone, white::Player)
    board[1,1,2] = (flat::Stone, black::Player)
    board[1,1,3] = (stand::Stone, white::Player)
    board[2,1,1] = (flat::Stone, black::Player)
    
    board
  end

  @testset "onehot <-> pos" begin
      @test onehot_to_pos(pos_to_onehot(FIELD_SIZE, FIELD_SIZE)) == (FIELD_SIZE, FIELD_SIZE)
      @test onehot_to_pos(pos_to_onehot(1, FIELD_SIZE)) == (1, FIELD_SIZE)
      @test onehot_to_pos(pos_to_onehot(FIELD_SIZE, 1)) == (FIELD_SIZE, 1)
      @test onehot_to_pos(pos_to_onehot(1, 1)) == (1, 1)
      @test onehot_to_pos(pos_to_onehot(2, 2)) == (2, 2)

      @test begin
          p = (rand(1:FIELD_SIZE), rand(1:FIELD_SIZE))
          idx_to_pos(pos_to_idx(p)) == p
      end
  end

  @testset "onehot <-> stone" begin
    @test onehot_to_stone(stone_to_onehot(flat::Stone)) == flat::Stone
    @test onehot_to_stone(stone_to_onehot(stand::Stone)) == stand::Stone
    if TakEnv.CAPSTONE_COUNT > 0
      @test onehot_to_stone(stone_to_onehot(cap::Stone)) == cap::Stone
    end
  end

  @testset "onehot <-> direction" begin
    @test onehot_to_direction(direction_to_onehot(north::Direction)) == north::Direction
    @test onehot_to_direction(direction_to_onehot(west::Direction)) == west::Direction
  end

  @testset "onehot <-> carry" begin
    @test onehot_to_carry(carry_to_onehot(west, possible_carries[end])) == (west, possible_carries[end])
    @test onehot_to_carry(carry_to_onehot(north, possible_carries[1])) == (north, possible_carries[1])
  end

  @testset "action <-> onehot" begin
    @test begin
        tmpaction = Action((1,1), flat::Stone, nothing, nothing, placement::ActionType)
        onehot_to_action(action_to_onehot(tmpaction)) == tmpaction
    end

    @test begin
        tmpaction = Action((1,1), stand::Stone, nothing, nothing, placement::ActionType)
        onehot_to_action(action_to_onehot(tmpaction)) == tmpaction
    end

    if TakEnv.CAPSTONE_COUNT > 0
      @test begin
        tmpaction = Action((1,1), cap::Stone, nothing, nothing, placement::ActionType)
        onehot_to_action(action_to_onehot(tmpaction)) == tmpaction
      end
    end

    @test begin
        tmpaction = Action((1,1), nothing, north::Direction, possible_carries[1], carry::ActionType)
        onehot_to_action(action_to_onehot(tmpaction)) == tmpaction
    end

    @test begin
        tmpaction = Action((rand(1:FIELD_SIZE), rand(1:FIELD_SIZE)), nothing, rand(possible_directions), rand(possible_carries), carry::ActionType)
        onehot_to_action(action_to_onehot(tmpaction)) == tmpaction
    end

    @test begin
        tmpaction = Action((TakEnv.FIELD_SIZE,TakEnv.FIELD_SIZE), nothing, west::Direction, possible_carries[end], carry::ActionType)
        onehot_to_action(action_to_onehot(tmpaction)) == tmpaction
    end
  end


  @testset "action <-> twohot" begin
    @test begin
        tmpaction = Action((1,1), flat::Stone, nothing, nothing, placement::ActionType)
        twohot_to_action(action_to_twohot(tmpaction)) == tmpaction
    end

    @test begin
      tmpaction = Action((1,1), stand::Stone, nothing, nothing, placement::ActionType)
      twohot_to_action(action_to_twohot(tmpaction)) == tmpaction
    end

    if TakEnv.CAPSTONE_COUNT > 0
      @test begin
          tmpaction = Action((1,1), cap::Stone, nothing, nothing, placement::ActionType)
          twohot_to_action(action_to_twohot(tmpaction)) == tmpaction
      end
    end

    @test begin
        tmpaction = Action((1,1), nothing, north::Direction, possible_carries[1], carry::ActionType)
        twohot_to_action(action_to_twohot(tmpaction)) == tmpaction
    end

    @test begin
        tmpaction = Action((rand(1:FIELD_SIZE), rand(1:FIELD_SIZE)), nothing, rand(possible_directions), rand(possible_carries), carry::ActionType)
        twohot_to_action(action_to_twohot(tmpaction)) == tmpaction
    end

    @test begin
        tmpaction = Action((TakEnv.FIELD_SIZE,TakEnv.FIELD_SIZE), nothing, west::Direction, possible_carries[end], carry::ActionType)
        twohot_to_action(action_to_twohot(tmpaction)) == tmpaction
    end
  end

  @testset "float_board" begin
    @test float_board(testboard())[1,1,TakEnv.FIELD_HEIGHT] == (stand::Stone, white::Player)
    @test float_board(testboard())[1,1,TakEnv.FIELD_HEIGHT-1] == (flat::Stone, black::Player)
    @test float_board(testboard())[1,1,1] === nothing
    @test unfloat_board(float_board(testboard())) == testboard()
    @test begin
        board, _, _ = random_game(30)
        unfloat_board(float_board(board)) == board
    end
  end

  @testset "onehot <-> piece" begin
    @test !any(piece_to_onehot(nothing))
    @test sum(piece_to_onehot((stand::Stone, white::Player))) == 1
    @test onehot_to_piece(piece_to_onehot((flat::Stone, white::Player))) == (flat::Stone, white::Player)
    @test onehot_to_piece(falses(Encoder.piece_encoding_length)) === nothing

    if TakEnv.CAPSTONE_COUNT > 0
      @test onehot_to_piece(piece_to_onehot((cap::Stone, white::Player))) == (cap::Stone, white::Player)
      @test onehot_to_piece(piece_to_onehot((cap::Stone, black::Player))) == (cap::Stone, black::Player)
      @test onehot_to_piece(vcat(falses(5),[true])) == (cap::Stone, black::Player)
    else
      @test onehot_to_piece(vcat(falses(3),[true])) == (stand::Stone, black::Player)
    end
  end

  @testset "onehot <-> toprow" begin
    @test sum(board_top_row_to_onehot(float_board(testboard()))) == 2
    @test length(board_top_row_to_onehot(float_board(testboard()))) == Encoder.piece_encoding_length*FIELD_SIZE*FIELD_SIZE
    @test begin
        board = float_board(random_game()[1])
        old_board = copy(board)
        enc = board_top_row_to_onehot(board)
        for x in 1:FIELD_SIZE
            for y in 1:FIELD_SIZE
                board[x,y,end] = (stand::Stone, black::Player)
            end
        end
        
        @assert old_board != board
        onehot_to_board_top_row!(board, enc)
        old_board == board
    end
  end

  @testset "onehot <-> player" begin
    @test player_to_onehot(white::Player) == [true, false]
    @test player_to_onehot(black::Player) == [false, true]
    #@test player_to_onehot(nothing) == [false, false]
    @test onehot_to_player(player_to_onehot(white::Player)) == white::Player
    @test onehot_to_player(player_to_onehot(black::Player)) == black::Player
    @test onehot_to_player(falses(2)) === nothing
  end


  @testset "onehot <-> stacks" begin
    @test sum(stacks_to_onehot(float_board(testboard()))) == 2
    @test begin
        board,_,_ = random_game()
        board = float_board(board)
        oldboard = copy(board)
        enc = stacks_to_onehot(board)
        for x in 1:FIELD_SIZE
            for y in 1:FIELD_SIZE
                for z in FIELD_HEIGHT-STACK_REPR_HEIGHT:FIELD_HEIGHT-1
                    board[x,y,z] = (cap::Stone, black::Player)
                end
            end
        end

        @assert board != oldboard
        onehot_to_stacks!(board, enc)
        board == oldboard
    end
    @test begin
        board = empty_board()
        for z in 1:FIELD_HEIGHT
            board[1,1,z] = (flat::Stone, white::Player)
        end

        sum(stacks_to_onehot(board)) == STACK_REPR_HEIGHT+1 # +1 for extra bit
    end
  end

  # Board to enc is only reversible if no stack is higher than STACK_REPR_HEIGHT
  @testset "enc <-> board" begin
    @test enc_to_board(board_to_enc(testboard())) == testboard()
    @test length(board_to_enc(random_game()[1])) == FIELD_SIZE*FIELD_SIZE*Encoder.piece_encoding_length+FIELD_SIZE*FIELD_SIZE+FIELD_SIZE*FIELD_SIZE*STACK_REPR_HEIGHT*2
  end

  @testset "compress/decompress board" begin
    @test begin
      for _ in 1:50
        board = random_game(30)[1]
        @assert decompress_board(compress_board(board)) == board
      end
      true
    end
  end

  @testset "random_encoded_game" begin
    @test !isnothing(random_encoded_game(nothing)[1])
  end

  @testset "rotation, mirroring" begin
    @test sum(Encoder.action_onehot_rotation_map) == sum(1:Encoder.action_onehot_encoding_length)

    board = TakEnv.random_game(10)[1]
    action_vec = reduce(.|, Encoder.action_to_onehot.(TakEnv.enumerate_actions(board, TakEnv.white::Player)))
    @test Encoder.rotate_action_vec(action_vec) != action_vec
    tmp_vec = Encoder.rotate_action_vec(action_vec)
    tmp_vec = Encoder.rotate_action_vec(tmp_vec)
    tmp_vec = Encoder.rotate_action_vec(tmp_vec)
    tmp_vec = Encoder.rotate_action_vec(tmp_vec)
    @test tmp_vec == action_vec

    action = TakEnv.Action((rand(1:TakEnv.FIELD_SIZE), rand(1:TakEnv.FIELD_SIZE)), TakEnv.flat::Stone, nothing, nothing, TakEnv.placement::ActionType)
    @test Encoder.rotate_action_vec(Encoder.action_to_onehot(action)) == Encoder.action_to_onehot(TakEnv.rotate_action(action))

    @test Encoder.mirror_action_vec(action_vec) != action_vec
    @test Encoder.mirror_action_vec(Encoder.mirror_action_vec(action_vec)) == action_vec
    @test Encoder.mirror_action_vec(Encoder.action_to_onehot(action)) == Encoder.action_to_onehot(TakEnv.mirror_action(action))


    board_rotated = TakEnv.rotate_board(board)
    action_vec_rotated = reduce(.|, Encoder.action_to_onehot.(TakEnv.enumerate_actions(board_rotated, TakEnv.white::Player)))
    @test Encoder.rotate_action_vec(action_vec) == action_vec_rotated

    board_mirrored = TakEnv.mirror_board(board)
    action_vec_mirrored = reduce(.|, Encoder.action_to_onehot.(TakEnv.enumerate_actions(board_mirrored, TakEnv.white::Player)))
    @test Encoder.mirror_action_vec(action_vec) == action_vec_mirrored
  end

  @testset "get_valid_moves" begin
    board = TakEnv.empty_board()
    @test sum(Encoder.get_valid_moves(board, TakEnv.white::Player)) > 0
    @test sum(Encoder.get_valid_moves(board, TakEnv.white::Player)) == size(TakEnv.enumerate_actions(board, TakEnv.white::Player), 1)
    @test size(Encoder.get_valid_moves(board, TakEnv.white::Player), 1) == action_onehot_encoding_length
  end

  @testset "board_to_conv_enc" begin
    board = testboard()
    enc = Encoder.board_to_conv_enc(board, TakEnv.white::Player)
    
    @test enc[1,1,1] == true
    @test enc[1,1,2] == true
    @test sum(enc) == 5 + TakEnv.FIELD_SIZE^2
    @test sum(Encoder.board_to_conv_enc(board, TakEnv.black::Player)) == 5

  end
end
end