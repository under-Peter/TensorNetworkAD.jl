using Base.Iterators: drop, take
using IterTools: iterated, imap

"""
    fixedpoint(f, guess, stopfun)

return the result of applying `guess = f(guess)`
until convergence. Convergence is decided by applying
`stopfun(guess)` which returns a Boolean.
"""
function fixedpoint(f, guess, stopfun)
    for state in iterated(f, guess)
        stopfun(state) && return state
    end
end

mutable struct StopFunction{T,S}
    oldvals::T
    counter::Int
    tol::S
    maxit::Int
end

"""
    (st::StopFunction)(state)
stopfunction for ctmrg, returning true if
singular values are converged or the maximum
number of iterations is reached.
"""
function (st::StopFunction)(state)
    st.counter += 1
    st.counter > st.maxit && return true

    vals = state[2]
    diff = norm(vals - st.oldvals)
    diff <= st.tol && return true
    st.oldvals = vals

    return false
end
