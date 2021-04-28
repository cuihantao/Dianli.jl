module Routines

using ..PowerSystem
using ..Dianli: THREAD_MODES
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra
using NLsolve: OnceDifferentiable, nlsolve

export nr_serial, nr_serial!, nr_threaded, nr_threaded!
export nr_jac_serial!, nr_jac_threaded!, run_nr, nlsolve_nr

include("./pflow.jl")

end
