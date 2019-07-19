using Base.Iterators: drop, take
using IterTools: iterated, imap

function fixedpoint(f, guess, init, stopfun)
    for state in iterated(x -> f(x,init), guess)
        stopfun(state) && return state
    end
end

mutable struct StopFunction{T,S}
    oldvals::T
    counter::Int
    tol::S
    maxit::Int
end


@Zygote.nograd StopFunction
function (st::StopFunction)(state)
    st.counter += 1
    st.counter > st.maxit && return true

    vals = state[3]
    diff = norm(vals - st.oldvals)
    diff <= st.tol && return true
    st.oldvals = vals

    return false
end

function fixedpointbackward(next, (c,t,vals), (a, χ, d))
    _, back = Zygote.forward(next,(c,t,vals),(a,χ,d))
    back1 = x -> back(x)[1]
    back2 = x -> back(x)[2]

    function backΔ(Δ)
        grad = back2(Δ)[1]
        for g in take(imap(back2,drop(iterated(back1, Δ),1)),100)
            grad .+= g[1]
            ng = norm(g[1])
            if ng < 1e-7
                break
            elseif ng > 10
                println("backprop not converging")
                # try to minimise damage by scaling to small
                grad ./= norm(grad)
                grad .*= 1e-4
                break
            end
        end
        (grad, nothing, nothing)
    end
    return backΔ
end

fixedpointAD(f, g, n, sf) = fixedpoint(f, g, n ,sf)

@Zygote.adjoint function fixedpointAD(f, guess, n, stopfun)
    r = fixedpoint(f, guess, n, stopfun)
    return r, Δ -> (nothing, nothing, fixedpointbackward(f, r, n)(Δ), nothing)
end
