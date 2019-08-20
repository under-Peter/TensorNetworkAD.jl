using TensorNetworkAD: fixedpoint

struct StopFunction{T,S}
    o::Ref{T}
    tol::S
end

mutable struct StopFunction2
    counter::Int
    limit::Int
end

@testset "fixedpoint" begin
    # example squareroot
    (st::StopFunction)(v) = abs(v - st.o[]) < st.tol ? true : (st.o[] = v; false)

    init = 9
    next(guess, n) = 1/2*(guess + n/guess)
    stopfun = StopFunction(Ref(Inf), 1e-9)

    #do nothing
    @test fixedpoint(x->next(x, 9), 9, x -> true) ≈ 9
    #evaluate once
    (st::StopFunction2)(v) = (st.counter += 1; st.counter == st.limit)
    @test fixedpoint(x->next(x, 9), 9, StopFunction2(0,2)) ≈ 1/2*(9 + 1)
end
