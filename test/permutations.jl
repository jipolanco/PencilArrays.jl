using OffsetArrays
using PencilArrays: Permutation, PermutedLinearIndices, PermutedCartesianIndices
using Test

@testset "Permuted indices" begin
    Aoff = OffsetArray(rand(3, 4, 5), -2, -1, -3)
    lin = LinearIndices(Aoff)
    cart = CartesianIndices(Aoff)

    perm = Permutation(3, 1, 2)
    plin = @inferred PermutedLinearIndices(lin, perm)
    pcart = @inferred PermutedCartesianIndices(cart, perm)

    for f in (size, axes)
        @test f(plin) == perm \ f(lin)
        @test f(pcart) == perm \ f(cart)
    end

    for n in LinearIndices(lin)
        @test lin[n] == plin[n]
        @test cart[n] == perm * pcart[n]
    end

    for I in CartesianIndices(lin)
        @test lin[I] == plin[perm \ I]
    end

    for J in CartesianIndices(plin)
        @test lin[perm * J] == plin[J]
    end

    # Iterate over permuted indices
    for n in plin
        @test n == lin[n] == plin[n]
        @test cart[n] == perm * pcart[n]
    end

    for J in pcart
        @test pcart[J] == J
    end

    for (I, J) in zip(cart, pcart)
        @test I == perm * J
    end
end
