from pyclbr import Function
from dolfin import *

mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), f"meshes/C25/parenchyma16.h5", "r")
hdf.read(mesh, "/mesh", False)



geo = mesh.ufl_cell()
P1 = FiniteElement('CG',geo,1)
ME = MixedElement(7*[P1])
V = FunctionSpace(mesh, "CG", 1)
p1 = Function(V)
p2 = Function(V)
p_xdmf = XDMFFile(MPI.comm_world, f"meshes/C25/results16/pvds/p_a.xdmf")
p_xdmf.read_checkpoint(p1, "p", 2)
p_xdmf.read_checkpoint(p2, "p", 20)

Volume = assemble(1*dx(mesh))
p_xdmf.close()
a = (1/133.322)*assemble(p1*dx(mesh))/Volume
b = (1/133.322)*assemble(p2*dx(mesh))/Volume
print(a-b)
print(a)
print(b)