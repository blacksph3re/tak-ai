
module TakUITest
using Test

using ..TakUI
using ..TakEnv

@testset "TakUI.jl" begin

  @testset "render" begin
    @test typeof(TakUI.render_board(TakEnv.random_game()[1])) == TakUI.Luxor.Drawing
  end

  if TakEnv.FIELD_SIZE == 5
    @testset "action_string" begin
      @test startswith(TakUI.action_string(TakEnv.Action((1,1), TakEnv.flat, nothing, nothing, TakEnv.placement)), "place")
      @test startswith(TakUI.action_string(TakEnv.Action((2, 1), nothing, TakEnv.west::Direction, (-1,0,0,0), TakEnv.carry)), "move")
      @test startswith(TakUI.action_string(TakEnv.Action((2, 1), nothing, TakEnv.west::Direction, (1,0,0,0), TakEnv.carry)), "move")
    end

    @testset "read_action" begin
      act = TakEnv.Action((1,1), TakEnv.flat, nothing, nothing, TakEnv.placement)
      @test TakUI.read_action(TakUI.action_string(act)) == act
      act = TakEnv.Action((2, 1), nothing, TakEnv.west::Direction, (-1,0,0,0), TakEnv.carry)
      @test TakUI.read_action(TakUI.action_string(act)) == act
      act = TakEnv.Action((2, 1), nothing, TakEnv.west::Direction, (1,0,0,0), TakEnv.carry)
      @test TakUI.read_action(TakUI.action_string(act)) == act
      @test TakUI.read_action("move (1, 2) east drop 1 2 3 4 5 6 7 8 9 1 2 3 4") === nothing
      @test TakUI.read_action("lorem ipsum dolor sit amet") === nothing
    end
  end
end

end