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
