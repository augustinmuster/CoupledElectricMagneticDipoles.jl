using Base
using LinearAlgebra
using BenchmarkTools
using DelimitedFiles
include("../../src/DDA.jl")
include("../../src/processing.jl")
#
global N=100
#benchmark parameters
N_samples=10
time=30
n_threads=Threads.nthreads()
# change default for `seconds` 
BenchmarkTools.DEFAULT_PARAMETERS.samples = N_samples
# change default for `seconds` 
BenchmarkTools.DEFAULT_PARAMETERS.seconds = time
#folder to store results
fold_name="benchmark_results"
if !isdir(fold_name)
    mkdir(fold_name)
end

function benchmark_CEMD(;fout=nothing,logout=stdout)
    #Number of function to benchmark
    N_functions=14
    median_time=zeros(N_functions)
    ###################################################################################################################
    #####                                           DDA.jl                                                         ####
    ###################################################################################################################
    println(logout,"##################################################")
    println(logout,"##################################################")
    println(logout,"N=",N)
    println(logout,"##################################################")
    println(logout,"##################################################")
    println(logout,"")
    println(logout,"")
    println(logout,"")
    println(logout,"##################################################")
    println(logout,"DDACore Module")
    println(logout,"##################################################")
    #load_dda_matrix_e
    println(logout,"*********************")
    println(logout,"load_dda_matrix_e")
    println(logout,"*********************")
    t=@benchmark DDACore.load_dda_matrix_e(rand(N,3),rand(ComplexF64,N),false)
    median_time[1]=median(t).time
    show(logout, MIME("text/plain"),t) 
    #load_dda_matrix_e_m
    println(logout,"")
    println(logout,"*********************")
    println(logout,"load_dda_matrix_e_m")
    println(logout,"*********************")
    t=@benchmark DDACore.load_dda_matrix_e_m(rand(N,3),rand(ComplexF64,N),rand(ComplexF64,N),false)
    median_time[2]=median(t).time
    show(logout, MIME("text/plain"),t) 
    #solve_system_e
    println(logout,"")
    println(logout,"*********************")
    println(logout,"solve equations e CPU")
    println(logout,"*********************")
    t= @benchmark DDACore.solve_system(rand(ComplexF64,3*N,3*N),rand(ComplexF64,3*N),"CPU",false) 
    median_time[3]=median(t).time
    show(logout, MIME("text/plain"),t) 
    println(logout,"")
    println(logout,"*********************")
    println(logout,"solve equations e GPU")
    println(logout,"*********************")
    t= @benchmark DDACore.solve_system(rand(ComplexF64,3*N,3*N),rand(ComplexF64,3*N),"GPU",false)
    median_time[4]=median(t).time
    show(logout, MIME("text/plain"),t) 
    #solve_system_e_m
    BLAS.set_num_threads(n_threads)
    println(logout,"")
    println(logout,"*********************")
    println(logout,"solve equations e m CPU")
    println(logout,"*********************")
    t=@benchmark DDACore.solve_system(rand(ComplexF64,6*N,6*N),rand(ComplexF64,6*N),"CPU",false)
    median_time[5]=median(t).time
    show(logout, MIME("text/plain"),t) 
    println(logout,"")
    println(logout,"*********************")
    println(logout,"solve equations e m GPU")
    println(logout,"*********************")
    t= @benchmark DDACore.solve_system(rand(ComplexF64,6*N,6*N),rand(ComplexF64,6*N),"GPU",false)
    median_time[6]=median(t).time
    show(logout, MIME("text/plain"),t) 
    #invert_matrix e 
    println(logout,"")
    println(logout,"*********************")
    println(logout,"invert matrix e CPU")
    println(logout,"*********************")
    t= @benchmark DDACore.solve_system(rand(ComplexF64,3*N,3*N),rand(ComplexF64,3*N),"CPU",false)
    median_time[7]=median(t).time
    show(logout, MIME("text/plain"),t) 
    println(logout,"")
    println(logout,"*********************")
    println(logout,"invert matrix e GPU")
    println(logout,"*********************")
    t= @benchmark DDACore.solve_system(rand(ComplexF64,3*N,3*N),rand(ComplexF64,3*N),"GPU",false)
    median_time[8]=median(t).time
    show(logout, MIME("text/plain"),t) 
    #invert_matrix e_m
    println(logout,"")
    println(logout,"*********************")
    println(logout,"invert matrix e m CPU")
    println(logout,"*********************")
    t=@benchmark DDACore.solve_system(rand(ComplexF64,6*N,6*N),rand(ComplexF64,6*N),"CPU",false)
    median_time[9]=median(t).time
    show(logout, MIME("text/plain"),t) 
    println(logout,"")
    println(logout,"*********************")
    println(logout,"invert ,matrix e m GPU")
    println(logout,"*********************")
    t= @benchmark DDACore.solve_system(rand(ComplexF64,6*N,6*N),rand(ComplexF64,6*N),"GPU",false)
    median_time[10]=median(t).time
    show(logout, MIME("text/plain"),t) 
    ###################################################################################################################
    #####                                           Processing.jl                                                  ####
    ###################################################################################################################
    println("")
    println(logout,"##################################################")
    println(logout,"PostProcessing Module")
    println(logout,"##################################################")
    #cross sections e
    println(logout,"")
    println(logout,"*********************")
    println(logout,"Cross sections e")
    println(logout,"*********************")
    t=@benchmark PostProcessing.compute_cross_sections_e(rand(),rand(N,3),rand(ComplexF64,N,3),rand(ComplexF64,N),rand(ComplexF64,N,3),verbose=false)
    median_time[11]=median(t).time
    show(logout, MIME("text/plain"),t) 
    #invert_matrix e_m
    println(logout,"")
    println(logout,"*********************")
    println(logout,"Cross sections e m")
    println(logout,"*********************")
    t=@benchmark PostProcessing.compute_cross_sections_e_m(rand(),rand(N,3),rand(ComplexF64,N,6),rand(ComplexF64,N),rand(ComplexF64,N),rand(ComplexF64,N,6),verbose=false)
    median_time[12]=median(t).time
    show(logout, MIME("text/plain"),t) 
    #invert_matrix e_m
    println(logout,"")
    println(logout,"*********************")
    println(logout,"LDOS e")
    println(logout,"*********************")
    t=@benchmark PostProcessing.ldos_e(rand(N,3), rand(ComplexF64), rand(ComplexF64,3*N,3*N), rand(10,3); dip=nothing)
    median_time[13]=median(t).time
    show(logout, MIME("text/plain"),t) 
    #invert_matrix e_m
    println(logout,"")
    println(logout,"*********************")
    println(logout,"LDOS e m")
    println(logout,"*********************")
    t=@benchmark PostProcessing.ldos_e_m(rand(N,3), rand(ComplexF64), rand(ComplexF64), rand(ComplexF64,6*N,6*N), rand(10,3); dip=nothing)
    median_time[14]=median(t).time
    show(logout, MIME("text/plain"),t) 
    ###################################################################################################################
    #####                                           Saving                                                         ####
    ###################################################################################################################
    if fout!==nothing
        writedlm(fout,[N transpose(median_time)])
    end
end

############## MAIN ################
fout=open(fold_name*"/res_"*string(n_threads)*"_cores.dat","w")
logout=open(fold_name*"/log_"*string(n_threads)*"_cores.dat","w")
writedlm(fout,["Load_e" "Load_e_m" "Solve_e_CPU" "Solve_e_GPU" "Solve_e_m_CPU" "Solve_e_m_GPU" "Inv_e_CPU" "Inv_e_GPU" "Inv_e_m_CPU" "inv_e_m_GPU" "CS_e" "CS_e_m" "LDOS_e" "LDOS_e_m"])
for i=7:12
    global N=Int(2^i)
    println("Numbers of dipoles: ",N)
    benchmark_CEMD(fout=fout,logout=logout)
end
close(fout)
close(logout)