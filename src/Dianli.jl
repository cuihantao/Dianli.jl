__precompile__()

module Dianli

using SparseArrays
using KLU
using NLsolve
using Andes
using PyCall: PyObject

import Base: convert

THREAD_MODES = Union{Type{Val{:serial}},Type{Val{:threaded}}}

include("basics.jl")
include("models/models.jl")
include("system.jl")
include("routines/routines.jl")
include("utils/utils.jl")

using .PowerSystem
using .Routines
using .BasicTypes

export System
export THREAD_MODES
export nr_serial, nr_threaded, nr_serial!, nr_threaded!
export nr_jac_serial!, nr_jac_threaded!, run_nr, nlsolve_nr
export Ymatrix, Ymatrix!
export convert

end
