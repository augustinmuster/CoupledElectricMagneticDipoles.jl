#imports
using DelimitedFiles
import CoupledElectricMagneticDipoles as CEMD

##################### INPUT FILE HERE ###################################
#file that contain all the refractive index for each frequency (wavelength(nm)/real part of n/imaginary part of n)
refractive_index="silicon_refractive_index.dat"
##########################################################################

#read lattice file
latt=CEMD.Geometries.gen_sphere_lattice_cubes(7,230e-9)

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
        L=CEMD.Alphas.depolarisation_tensor(latt[j,6],latt[j,6],latt[j,6],latt[j,7])
        epsilon=(real_eps[Int(latt[j,5])]+im*imag_eps[Int(latt[j,5])])^2
        alpha0[j,:,:]=CEMD.Alphas.alpha_0(epsilon,1,L,latt[j,7])
        alpha[j,:,:]=CEMD.Alphas.alpha_radiative(alpha0[j,:,:],knorm)
    end

    #println(real_eps,imag_eps)
    p,e_inc=CEMD.DDACore.solve_DDA_e(knorm,latt[:,1:3],alpha,CEMD.InputFields.plane_wave,solver="AUTO")
    #compute cross section
    res[i,:]=CEMD.PostProcessing.compute_cross_sections(knorm,latt[:,1:3],p,e_inc,alpha0)
end

#write results to file
fout=open("results.dat","w")
writedlm(fout,res)
close(fout)
