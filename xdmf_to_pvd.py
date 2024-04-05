from pyclbr import Function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from IPython import embed
import argparse

# Command line arguments checking for different simulation experiments
parser = argparse.ArgumentParser()
parser.add_argument("--order", type=int, default=1)
parser.add_argument("--variation", type=str, default="base")
parser.add_argument("-res", "--resolution", type=int, default=16)
args = parser.parse_args()

mpi_comm = MPI.comm_world
res = args.resolution
case = args.variation
patient = "C0"

if case.lower() == "base":
    num_runs = 1
elif case.lower() == "var1":
    num_runs = 3
elif case.lower() == "var2":
    num_runs = 3
elif case.lower() == "var3":
    num_runs = 2
elif case.lower() == "var4":
    num_runs = 5
elif case.lower() == "var5":
    num_runs = 3
else: 
    print("Unknown experiment.")
    sys.exit()


main_dir = f"results/{patient}/"
if case.lower() == "base":
    in_dir_names = [main_dir + case.lower() + "/"]
elif case.lower() == "var1":
    in_dir_names = [f"{main_dir}/base/", f"{main_dir}/Res_FaghihSharp/", f"{main_dir}/Res_Pizzo/"]
elif case.lower() == "var2":
    in_dir_names = [f"{main_dir}/base/", f"{main_dir}/high_DS_resistance/", f"{main_dir}/low_DS_resistance/"]
elif case.lower() == "var3":
    in_dir_names = [f"r{main_dir}/constant_filtration/", f"{main_dir}/nonconstant_filtration/"]
elif case.lower() == "var4":
    in_dir_names = [f"{main_dir}/transfer_case{i}" for i in range(num_runs)]
    in_dir_names[0] = "base"
elif case.lower() == "var5":
    in_dir_names = [f"results/{patient}/base/", f"results/{patient}/high_kECS/", f"results/{patient}/very_high_kECS/"]

# Read XDMF File
def read_xdmf(infile, mesh, order, tag):
    
    V = FunctionSpace(mesh, "CG", order)
    p = Function(V)

    p_xdmf = XDMFFile(MPI.comm_world, infile)
    p_xdmf.read_checkpoint(p, "p", tag)
    p_xdmf.close()
   
    return p

mesh = Mesh()

comps = ["a", "c", "v", "pa", "pc", "pv", "e"]
compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
Nc = len(comps)

hdf = HDF5File(mesh.mpi_comm(), f"meshes/C0/parenchyma{res}.h5", "r")
hdf.read(mesh, "/mesh", False)

for j, variation in enumerate(in_dir_names):
    Files = [File(f"{variation}/pvds/p_{comp}.pvd") for comp in comps]
    for k in range(6):
        for i, comp in enumerate(comps):
            target = variation + f"pvds/p_{comp}.xdmf"
            pxdmf = read_xdmf(target, mesh, args.order, k)
            Files[i] << pxdmf

[File_.close for File_ in Files]