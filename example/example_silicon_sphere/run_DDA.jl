#imports
using DelimitedFiles
include("../../DDA.jl")
include("../../alpha.jl")
include("../../input_fields.jl")
include("../../processing.jl")
##################### INPUT FILES HERE ###################################
refractive_index="silicon_refractive_index.dat" #file that contain all the refractive index for each frequency (wavelength(nm) |tab| real part of n |tab| imaginary part of N)
lattice="sphere_lattice.dat" #file that contain the lattice (x coordinate |tab| y coordinate |tab| z coordinate |tab| distance from the origine |tab| polarisability a0 tensor (without radiative correction))
##########################################################################
#read lattice file

latt_file=open(lattice,"r")
latt=readdlm(latt_file,'\t',Float64,'\n')

#run the DDA for all line of the refractive index files
n_file=open(refractive_index,"r")
ref_id=readdlm(n_file,'\t',Float64,'\n')
res=zeros(Float64,length(ref_id[:,1]),4)
for i=1:length(ref_id[:,1])
    #read frequency+espilon
    freq=ref_id[i,1]*1e-9
    real_eps=[]
    imag_eps=[]
    for j=2:length(ref_id[i,:])
        if j%2==0
            append!(real_eps,ref_id[i,j])
        else
            append!(imag_eps,ref_id[i,j])
        end
    end
    #norm of the wave vector
    knorm=2*pi/freq
    #generate polarisabilities
    n=length(latt[:,1])

    alpha=zeros(ComplexF64,n,3,3)
    alpha0=zeros(ComplexF64,n,3,3)
    for j=1:n
        L=depolarisation_tensor(latt[j,6],latt[j,6],latt[j,6],latt[j,7])
        epsilon=(real_eps[Int(latt[j,5])]+im*imag_eps[Int(latt[j,5])])^2
        alpha0[j,:,:]=alpha_0(epsilon,1,L,latt[j,7])
        alpha[j,:,:]=alpha_radiative(alpha0[j,:,:],knorm)
    end

    #println(real_eps,imag_eps)
    p,e_inc=solve_DDA(knorm,latt[:,1:3],alpha,plane_wave,solver="LU")
    #compute cross section
    res[i,:]=compute_cross_sections(knorm,latt[:,1:3],p,e_inc,alpha0)
end

#write results to file
fout=open("results.dat","w")
writedlm(fout,res)
close(fout)
