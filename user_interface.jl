
mutable struct Simulation
    knorms::Array{Float64,1}
    eps_h::Float64
    Nl::Int64
    #constructor
    function Simulation(knorms::Array{Float64,1},eps_h::Float64)
        this=new()
        this.knorms=knorms
        this.eps_h=eps_h
        this.Nl=length(knorms)
        return this
    end
end

mutable struct EDipole
    sim::Simulation
    epsilon::ComplexF64
    position::Array{Float64,1}
    volume::Float64
    compute_alpha::Function
    alpha::Array{ComplexF64,1}
    #constructor
    function EDipole(sim::Simulation,epsilon::ComplexF64,position::Array{Float64,1},volume::Float64,compute_alpha::Function)
        this=new()
        this.sim=sim
        this.epsilon=epsilon
        this.position=position
        this.volume=volume
        this.compute_alpha=compute_alpha
        this.alpha=zeros(ComplexF64,this.sim.Nl)
        for i=1:this.sim.Nl
            this.alpha[i]=compute_alpha(this.sim.knorms[i],this.epsilon,this.sim.eps_h,this.volume)
        end
        return this
    end
end

mutable struct EMDipole
    sim::Simulation
    epsilon::ComplexF64
    position::Array{Float64,1}
    volume::Float64
    compute_alpha_em::Function
    alpha_e::Array{ComplexF64,1}
    alpha_m::Array{ComplexF64,1}
    #constructor
    function EMDipole(sim::Simulation,epsilon::ComplexF64,position::Array{Float64,1},volume::Float64,compute_alpha_em::Function)
        this=new()
        this.sim=sim
        this.epsilon=epsilon
        this.position=position
        this.volume=volume
        this.compute_alpha_em=compute_alpha_em
        this.alpha_e=zeros(ComplexF64,this.sim.Nl)
        this.alpha_m=zeros(ComplexF64,this.sim.Nl)
        for i=1:this.sim.Nl
            ae,am=compute_alpha_em(this.sim.knorms[i],this.epsilon,this.sim.eps_h,this.volume)
            this.alpha_e[i]=ae
            this.alpha_m[i]=am
        end
        return this
    end
end

struct ESystem
end


function alpha_t(knorm,eps,eps_h,volume)
    return 2*eps/eps_h*volume/knorm^3
end

function alpha_t_em(knorm,eps,eps_h,volume)
    return 2*eps/eps_h*volume/knorm^3,4*eps/eps_h*volume/knorm^3
end


sim=Simulation([1.,2.,3.],12.)
println(sim.knorms)
println(sim.eps_h)
println(sim.Nl)

dip=EDipole(sim,2. +0. *im,[1.,2.,3.],19e-4,alpha_t)
println(dip.alpha)

dipem=EMDipole(sim,2. +0. *im,[1.,2.,3.],19e-4,alpha_t_em)
println(dipem.alpha_e)
println(dipem.alpha_m)