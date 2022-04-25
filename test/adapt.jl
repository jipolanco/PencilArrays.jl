using Adapt
using MPI
using PencilArrays
using Test

include("include/jlarray.jl")
using .JLArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
rank == 0 || redirect_stdout(devnull)

pen = Pencil((12, 43), comm)
u = PencilArray(pen, rand(Float64, size_local(pen)...))

@testset "Adapt" begin
    @testset "Float64 -> Float32" begin
        @assert u isa PencilArray{Float64, 2}
        @assert parent(u) isa Array{Float64, 2}
        v = @inferred adapt(Array{Float32}, u)
        @test v isa PencilArray{Float32, 2}  # wrapper type is preserved
        @test parent(v) isa Array{Float32, 2}
        @test u ≈ v
    end

    # Try changing array type (Array -> JLArray)
    @testset "Array -> JLArray" begin
        @assert Pencils.typeof_array(u) === Array

        v = @inferred adapt(JLArray, u)
        @test v isa PencilArray{Float64, 2}
        @test Pencils.typeof_array(v) === JLArray
        @test parent(v) isa JLArray{Float64, 2}
        @test JLArray(parent(u)) == parent(v)

        # Similar but changing element type
        v = @inferred adapt(JLArray{Float32}, u)
        @test v isa PencilArray{Float32, 2}
        @test Pencils.typeof_array(v) === JLArray
        @test parent(v) isa JLArray{Float32, 2}
        @test JLArray(parent(u)) ≈ parent(v)
    end
end
