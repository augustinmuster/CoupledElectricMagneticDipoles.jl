#!/bin/bash

julia --threads=1 benchmark_CEMD.jl
julia --threads=2 benchmark_CEMD.jl
julia --threads=4 benchmark_CEMD.jl
julia --threads=8 benchmark_CEMD.jl
julia --threads=16 benchmark_CEMD.jl
