using StaticArrays
using Parameters

"""
    UnitCell{TA}
holds:
    - tensors of type `TA` in a dictionary (field `tensors`)
    - lattice vectors (field `lvecs`)
    - transformation matrix to map cartesian coordinates into the unitcell (field `m`)
"""
struct UnitCell{TA}
    tensors::Dict{Tuple{Int,Int}, TA}
    lvecs::NTuple{2,SVector{2,Int}}
    m::SArray{Tuple{2,2},Float64,2,4}
    UnitCell(ts, lvecs) = new{eltype(values(ts))}(ts,lvecs, inv(hcat(lvecs...)))
end

Base.eltype(::UnitCell{TA}) where TA = TA

Base.similar(uc::UnitCell) = UnitCell(typeof(uc.tensors)(), uc.lvecs)


"""
    shiftcoordinates(v, m, (l1,l2))
move from cartesian coordinates `v` to unitcell coordinates described by
lattice vectors `l1`,`l2`, transformation matrix `m`.
"""
function shiftcoordinates(v::SVector{2}, m, (l1,l2))
    s1, s2 = round.(Int,m * v, RoundDown)
    x, y = v - s1*l1 - s2*l2 + one.(v)
    return x,y
end

function Base.getindex(uc::UnitCell, x::T, y::T) where T <: Integer
    @unpack m, lvecs, tensors = uc
    x, y = shiftcoordinates(SVector{2,T}(x,y), m, lvecs)
    return tensors[(x,y)]
end

function Base.setindex!(uc::UnitCell, A, x::T, y::T) where T <: Integer
    @unpack m, lvecs, tensors = uc
    x, y = shiftcoordinates(SVector{2,T}(x,y), m, lvecs)
    tensors[(x,y)] = A
    return uc
end
