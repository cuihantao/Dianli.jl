__precompile__()

module Dian

using SparseArrays
using NLsolve
using Andes
using PyCall: PyObject

import Base: convert

THREAD_MODES = Union{Type{Val{:serial}},Type{Val{:threaded}}}

include("basics.jl")
include("models/models.jl")
include("system.jl")
include("routines/routines.jl")

using .PowerSystem
using .Routines
using .BasicTypes

export System
export nr_serial, nr_threaded
export nr_serial!, nr_threaded!
export nr_cb_serial!, nr_cb_threaded!
export convert

end
