module TakInterfaceTest

using AlphaZero
using Test

using ..TakEnv
using ..TakInterface


@testset "TakInterface.jl" begin
  @testset "AlphaZero Test scripts" begin
    @test isnothing(AlphaZero.Scripts.test_game(TakInterface.TakSpec()))
  end
end

end