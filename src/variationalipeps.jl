function energy(h, t, χ, tol, maxit)
    d = size(t,1)
    ap = einsum(((1,2,3,4,5),(-1,-2,-3,-4,-5)), (t, t'), (1,-1,2,-2,3,-3,4,-4,5,-5))
    ap = reshape(ap, d^2, d^2, d^2, d^2, size(t,5), size(t,5))
    a = einsum(a,(1,2,3,4,-5,-5), (1,2,3,4))
    c, t, vals = ctmrg(a, χ, tol, maxit)

    return expectationvalue(h, ap, t, e)
end

function expectationvalue(h, ap, t, e)
    l = einsum(
        ((1,-6,-1),(-1,-2),(-2,-5,-3),(-3,-4),(-4,-7,5),(-6,2,-7,5,3,4)),
        (t,c,t,c,t,a),
        (1,2,3,4,5))
    norm = einsum(((1,2,3,4,5),(1,2,3,4,5)), (l,l),())[]
    e = einsum(((1,2,-3,-4,5),(1,2,-5,-6,5), (-3,-4,-5,-6)), (l,l,h),())[]
    return e/norm
end

function optimiseipeps(t, χ, d, maxit)
    let energy = x -> energy(h, x, χ, tol, maxit)
        res = optimize(energy, Δ -> Zygote.gradient(energy,Δ)[1],
                        t, LBFGS(), inplace = false)
    end
    return res.minimizer
end
