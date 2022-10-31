#imports
using DelimitedFiles
include("../../src/DDA.jl")
include("../../src/alpha.jl")
include("../../src/input_fields.jl")
include("../../src/processing.jl")

##################### INPUT FILES HERE ###################################
refractive_index="refractive_index.dat" #file that contain all the refractive index for each frequency (wavelength(nm) |tab| real part of n |tab| imaginary part of N)
lattice="pos.dat" #file that contain the lattice (x coordinate |tab| y coordinate |tab| z coordinate |tab| distance from the origine |tab| polarisability a0 tensor (without radiative correction))
##########################################################################
alpha_out=open("electric_field.dat","w")
#read lattice file

latt_file=open(lattice,"r")
latt=readdlm(latt_file)

#latt=[0 0 0 0 1 230e-9 230e-9^3*4/3*pi]
#latt=[0 0 0 0 1 230e-9 0;920e-9 0 0 920e-9 1 230e-9 0]
#latt=[-79e-9 0 0 79e-9 1 75e-9 0;119e-9 0 0 119e-9 1 115e-9 0]
#run the DDA for all line of the refractive index files
n_file=open(refractive_index,"r")
ref_id=readdlm(n_file,'\t',Float64,'\n')
res=zeros(Float64,length(ref_id[:,1]),4)
for i=1:length(ref_id[:,1])
    #read frequency+espilon
    freq=ref_id[i,1]
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
    alpha_e=zeros(ComplexF64,n,3,3)
    alpha_m=zeros(ComplexF64,n,3,3)
    for j=1:n
        #radius of the particle
        rad=1
        refr_id=1
        ae,am=alpha_e_m_mie_renorm(knorm,rad,real_eps[refr_id]+im*imag_eps[refr_id],1)
        alpha_e[j,:,:]=copy(ae*[1 0 0;0 1 0;0 0 1])
        alpha_m[j,:,:]=copy(am*[1 0 0;0 1 0;0 0 1])
    end
    p,m,e_inc,h_inc,e_inp,h_inp=solve_DDA_e_m(knorm,latt[:,1:3],alpha_e,alpha_m,plane_wave_e_m_renorm)
    #compute cross section
    if i==1
        fields=zeros(Float64,n,12)
        for k=1:n
            fields[k,1]=real(e_inc[k,1])
            fields[k,2]=imag(e_inc[k,1])
            fields[k,3]=real(e_inc[k,2])
            fields[k,4]=imag(e_inc[k,2])
            fields[k,5]=real(e_inc[k,3])
            fields[k,6]=imag(e_inc[k,3])
            fields[k,7]=real(h_inc[k,1])
            fields[k,8]=imag(h_inc[k,1])
            fields[k,9]=real(h_inc[k,2])
            fields[k,10]=imag(h_inc[k,2])
            fields[k,11]=real(h_inc[k,3])
            fields[k,12]=imag(h_inc[k,3])
        end
        writedlm(alpha_out,fields)
    end
    res[i,:]=compute_cross_sections_e_m(knorm,latt[:,1:3],p,m,e_inc,h_inc,e_inp,h_inp,alpha_e,alpha_m)
end

#write results to file
fout=open("results.dat","w")
writedlm(fout,res)
close(fout)

close(alpha_out)