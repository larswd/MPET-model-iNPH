from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython import embed
from analyse_data import read_data

#os.system("dijitso clean")
patient = "C25"
resolutions = [8,16,32,64, 96] 
surf_area = np.zeros(len(resolutions))
first_run = True


for i,res in enumerate(resolutions):    
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), f"meshes/partial_{patient}/parenchyma{res}.h5", "r")
    hdf.read(mesh, "/mesh", False)
    SD = MeshFunction("size_t", mesh,mesh.topology().dim())
    hdf.read(SD, "/subdomains")
    lookup_table = MeshFunction("size_t", mesh,mesh.topology().dim())
    hdf.read(lookup_table, "/lookup_table")
    bnd = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
    hdf.read(bnd, "/boundaries")

    ds = Measure('ds')(subdomain_data=bnd)
    dS = Measure('dS')(subdomain_data=bnd)
    dx = Measure('dx')(subdomain_data=SD)
    surf_area[i] = assemble(1*ds(mesh))
    print(surf_area)
    os.system(f"python3 MPET_brain.py -T 1800 -dt 25 -case partial_{patient} -res {res} -Q_in 712.15 --skip_pvds")
    if first_run:
        legends, p_matrix = read_data("partial_C25",res)
        p_avgs = np.zeros((len(resolutions),len(p_matrix[:,0]), len(p_matrix[0,:])))
        p_avgs[0,:,:] = p_matrix
        first_run = False
    else:
        legends, p_matrix = read_data("partial_C25", res)
        p_avgs[i,:,:] = p_matrix


plt.plot(resolutions, surf_area)
plt.xlabel("Mesh resolution")
plt.ylabel("Surface area")
plt.title("Mesh surface area for C25 as function of mesh resolution")
plt.show()

for i in range(1,8):
    for j in range(len(resolutions)):
        plt.plot(p_avgs[0,0,:], p_avgs[j,i,:])
    plt.xlabel("Time [s]")
    plt.ylabel("Average pressure [mmHg]")
    plt.title(f"Average pressure in {legends[i]} for different resolutions")
    plt.legend(["Res = 8", "Res = 16", "Res = 32", "Res = 64", "Res = 96"])
    plt.show()