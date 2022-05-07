
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os

#os.system("dijitso clean")
# Leser kommmandolinjeargumenter

start = time.time()

T = 2800
dt = 20

patient = sys.argv[1]
res = int(sys.argv[2])
order = int(sys.argv[3])
Q_in = float(sys.argv[4])


parameters["krylov_solver"]["maximum_iterations"] = 5000
parameters["krylov_solver"]["monitor_convergence"] = True
#parameters["krylov_solver"]["relative_tolerance"] = 2e-5

#os.system(f"mkdir meshes/{args.patient}/results{args.resolution}")
#os.system(f"mkdir meshes/{args.patient}/results{args.resolution}/pvds")
#sol_pa = File(f"meshes/{args.patient}/results{args.resolution}/pvds/p_a.pvd")
#sol_pc = File(f"meshes/{args.patient}/results{args.resolution}/pvds/p_c.pvd")
#sol_pv = File(f"meshes/{args.patient}/results{args.resolution}/pvds/p_v.pvd")
#sol_pe = File(f"meshes/{args.patient}/results{args.resolution}/pvds/p_e.pvd")
#sol_ppa = File(f"meshes/{args.patient}/results{args.resolution}/pvds/p_pa.pvd")
#sol_ppc = File(f"meshes/{args.patient}/results{args.resolution}/pvds/p_pc.pvd")
#sol_ppv = File(f"meshes/{args.patient}/results{args.resolution}/pvds/p_pv.pvd")

mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), f"meshes/{patient}/parenchyma{res}.h5", "r")
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
os.system(f'echo "File read" >> log_{patient}.txt')

#List of measures to circumvent FEniCS integration bug
dxs = [dx(domain=mesh, subdomain_data=SD, subdomain_id=i) for i in range(1,4)]

mpi_comm = MPI.comm_world

geo = mesh.ufl_cell()
h = mesh.hmin()

os.system(f'echo "Testing Volumes">> log_{patient}.txt')

compartments = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
Nc = len(compartments)

brain_regions = [1,2,3]
os.system(f'echo "{brain_regions}" >> log_{patient}.txt')

P1 = FiniteElement('CG',geo,order)
ME = MixedElement(Nc*[P1])
V = FunctionSpace(mesh,ME)
p = TrialFunctions(V)
q = TestFunctions(V)

R = FunctionSpace(mesh, P1)

# PHYSICAL PARAMETERS (hentet fra Vegards skript)
# FLUID
rho_f = Constant(1./1000)		# g/(mm^3)
nu_f = Constant(0.658)			# mm**2/s
mu_f = Constant(nu_f*rho_f)		# g/(mm*s)

mu_b = 3*mu_f


r_a = 0.000939857688751   # mmHg/mL/min    these two based on our calculation of resistance from F&S for vessels (not PVS)
r_v = 8.14915973766e-05   # mmHg/mL/min    script: /Waterscape/Waterscape/PhD/paperIV/Stoverud/compartment-ode-model 

r_factor = 133*1e6*60     # mmHg/(mL/min) = 133 Pa/(1e-3 L / 60 s) = 133 Pa / (1e-3 * 1e-3 m^3/ 60s) = 133*1e6*60 Pa/s
r_ECS = 0.57*r_factor     # Pa/(m^3/s)    (from ECS permeability of Karl-Erik together with the 2D geometry by Adams)

r_IEG_v = 0.64*r_factor    # venous inter-endfeet gaps
r_IEG_a = 0.57*r_factor    # arterial inter-endfeet gaps
r_cv = 125*r_factor         # resistance over capillary wall
r_pa = 1.02*r_factor        # resistance along periarterial spaces
r_pv = 0.079*r_factor       # resistance along perivenous spaces
r_pc = 32.24*r_factor       # resistance along pericapillary spaces


k_ECS = 20*1e-18/mu_f          # m^2/(Pa*s)
constant = r_ECS*k_ECS/mu_f # [1/m]

k_a = 1e6*constant*mu_b/(r_a*r_factor)     # constant is SI, mu_b is SI, r_a is SI thus k_a is SI [m^2s/Pa]
k_c = 1e6*1.44e-15/mu_b
k_v = 1e6*constant*mu_b/(r_v*r_factor)
k_pa = 1e6*constant*mu_f/r_pa
k_pc = 1e6*constant*mu_f/r_pc
k_pv = 1e6*constant*mu_f/r_pv
k_ECS = 1e6*k_ECS


kappas = [Constant(k_a), Constant(k_c), Constant(k_v), Constant(k_pa), Constant(k_pc), Constant(k_pv), Constant(k_ECS)]
os.system(f'echo "kappas  {[float(kappas[i]) for i in range(len(kappas))]}" >> log_{patient}.txt')
#[3.63e-8, 1.44e-9, 1.13e-6, 2e-11]
n = FacetNormal(mesh)

Qprod = Expression("5.787e-5", degree=0)

t = 0

beta1 = 1e-3 # [1/(Pa*s)]  periarterial inflow
beta2 = 1e-3   # Venous outflow
beta3 = 1e-7   # perivenous outflow


CSF_prod = 0.33*1e3/60.
CSF_infusion = 1.5*1e3/60.
CSF_in = CSF_prod+CSF_infusion
os.system(f'echo "{1e-9*CSF_in}" >> log_{patient}.txt')

ventricle_area = assemble(interpolate(Constant(1), R)*ds(2))
surface_area = assemble(interpolate(Constant(1), R)*(ds(1)+ ds(2)+ds(3)))
pia_area = assemble(interpolate(Constant(1), R)*ds(1))

Volume = assemble(1*dx(mesh))

with open(f"meshes/{patient}/patdat.txt", 'r') as infile:
    line = infile.readline().split()
    Rout = float(line[1])
    line = infile.readline().split()
    blood_flow = float(line[1])
    line = infile.readline().split()
    p0 = float(line[1])


#MPI.barrier(MPI.comm_world)
b_avg = blood_flow*1e3/60./surface_area # mL/min = cm^3/min = 1e3*mm^3/(60s)
b_in = Constant(b_avg)

w_a_c = b_avg*surface_area/(Volume*60*133.33)   # For a ~60 mmHg drop from average Arterial to Capillary Pressure
w_c_v = b_avg*surface_area/(Volume*12.5*133.33) # For a ~12.5 mmHg drop from average Capillary to Venous pressure
w_pc_c = (Volume*1e-9*r_cv)**-1
w_pv_v = 1e-15 # flow into veins?   Justert opp. Muligens for stor
w_pa_a = 1e-15 # flow from arteries to arterial PVS? Justert opp. Muligens for stor
w_pa_pc = 1e-6
w_pc_pv = 1e-6
w_pa_pv = 0
w_a_pv  = 0
w_a_pc  = 0
w_a_v   = 0
w_c_pa  = 0
w_c_pv  = 0
w_v_pa  = 0 
w_v_pc  = 0

w_e_a = 0
w_c_e = 0
w_e_v = 0
w_pv_e = (Volume*1e-9*r_IEG_v)**-1    # Omega is 1/(V*R) ~ 1/(Pa*s/m^3*m^3) = 1/(Pa*s) 
w_pa_e = (Volume*1e-9*r_IEG_a)**-1  

w_pc_e = 1e-10 # flow from capillary PVS to extracellular matrix ?


w = np.array([[0, w_a_c, w_a_v, w_pa_a, w_a_pc, w_a_pv, w_e_a],
              [w_a_c, 0, w_c_v, w_c_pa, w_pc_c, w_c_pv, w_c_e], 
              [w_a_v, w_c_v, 0, w_v_pa, w_v_pc, w_pv_v, w_e_v],
              [w_pa_a, w_c_pa, w_v_pa, 0, w_pa_pc, w_pa_pv, w_pa_e],
              [w_a_pc, w_pc_c, w_v_pc, w_pa_pc, 0, w_pc_pv, w_pc_e],
              [w_a_pv, w_c_pv, w_pv_v, w_pa_pv, w_pc_pv, 0, w_pv_e],
              [w_e_a, w_c_e, w_e_v, w_pa_e, w_pc_e, w_pv_e, 0]])

os.system(f'echo "Initialised transfers" >> log_{patient}.txt')



def Compliance(p,E = 0.2e6, p0 = p0*133.33):
    if p < 13*13.322:
        if p0 > 13*133.322:
            p0 = 12*133.322
        return 1./(E*(13*133.322 - p0))
    else:
        if p0 > p:
            p0 = p - 2*133.322
        return 1./(E*(p-p0))

C_array = [1e-8, 1e-8, 1e-8, Constant(1e-8), Constant(1e-8), Constant(1e-8), 1e-8]
p_AG = 8.4*133.33           # AG pressure
p_CSF = Constant(10*133.33)  # CSF pressure
p_crib = Constant(0)        # cribriform plate pressure

# Initialbetingelse
p_1 = interpolate(Expression(("9.2*133.322", "9.2*133.322", "9.2*133.322", "9.2*133.322", "9.2*133.322", "9.2*133.322", "9.2*133.322"), degree=1), V)


os.system(f'echo "starting bilinear generation" >> log_{patient}.txt')
# Bilineær form
F = 0

for i in range(Nc):
	F += kappas[i]*inner(grad(p[i]),grad(q[i]))*dx
	F +=  C_array[i]/dt*inner(p[i]-p_1[i],q[i])*dx
	for j in range(Nc):
		if i != j:
			#print i,j
			F += Constant(w[i][j])*inner(p[i]-p[j],q[i])*dx


# Neumann BCS
# compartments = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]

F -= b_in*q[0]*(ds(1))#+ds(2)+ds(3))      # ARTERIAL INFLOW BC1 ds(1) pial, ds(2) ventricles

#F += CSF_prod/ventricle_area*q[1]*ds(2) # Choroid plexus production. 

F -= beta1*(p_CSF - p[3])*q[3]*ds(1)    # CSF-to-PVS-transfer modeled by ODE
#F -= beta2*(p_CSF - p[3])*q[3]*ds(2)
#F -= beta2*(p_CSF - p[3])*q[3]*ds(3)

F -= beta2*((p_AG+p_CSF)/2 - p[2])*q[2]*ds(1)          # Venous outflow
F -= beta3*((p_AG+p_CSF)/2 - p[5])*q[5]*ds(1)     # venous-PVS-to-lymph transfer, pressure = 8.4

# Dirichlet BC
p_v_pia = Expression("8.44*133.32",degree=1, t=t)
bcp = DirichletBC(V.sub(2), p_v_pia, bnd, 1)

bcu = []#[bcp]


os.system(f'echo "assembling" >> log_{patient}.txt')
A = assemble(lhs(F))
p_ = Function(V)

t = 0


alpha = (Rout + 1.5294)/11.0098

os.system(f'echo "{alpha}" >> log_{patient}.txt')
R_AG = 10.81*r_factor*alpha
R_crib = 67*r_factor*alpha


if int(mpi_comm.rank) == 0:
    os.system(f"mkdir meshes/{patient}/results{res}/")
    pressure_averages = open(f"meshes/{patient}/results{res}/pressure_average.dat", 'w+')
    pressure_averages.write("Time Arterial Capillary Venous PVS_a PVS_c PVS_v ISF\n")
    out = [open(f"meshes/{patient}/results{res}/"+'%d.txt'%region,'w') for region in brain_regions]
    [out[i].write("Time Arterial Capillary Venous PVS_a PVS_c PVS_v ISF\n") for i in range(len(brain_regions))]

region_volumes = [assemble(1*dx(region, domain=mesh)) for region in brain_regions]

os.system(f'echo "{region_volumes}" >> log_{patient}.txt')

P = np.zeros(7)

p_mmHg = Function(V)
p_mmHg.vector()[:] = p_1.vector()/133.322
k = 0

sol_pa = XDMFFile(MPI.comm_world, f"meshes/{patient}/results{res}/pvds/p_a.xdmf")
sol_pc = XDMFFile(MPI.comm_world, f"meshes/{patient}/results{res}/pvds/p_c.xdmf")
sol_pv = XDMFFile(MPI.comm_world, f"meshes/{patient}/results{res}/pvds/p_v.xdmf")
sol_pe = XDMFFile(MPI.comm_world, f"meshes/{patient}/results{res}/pvds/p_e.xdmf")
sol_ppa = XDMFFile(MPI.comm_world, f"meshes/{patient}/results{res}/pvds/p_pa.xdmf")
sol_ppc = XDMFFile(MPI.comm_world, f"meshes/{patient}/results{res}/pvds/p_pc.xdmf")
sol_ppv = XDMFFile(MPI.comm_world, f"meshes/{patient}/results{res}/pvds/p_pv.xdmf")


tag = 0

while t < T:
    
    p_v_pia.t = t
    b = assemble(rhs(F))
    
    # ODE for nytt trykk. Eksplisitt diskretisering.
    if k > 29:
        p_CSF_new = float(p_CSF) + dt/Compliance(float(p_CSF))*(CSF_in*1e-9 - 1/R_AG*(float(p_CSF) - p_AG) - 1/R_crib*(float(p_CSF) - p_crib))
        p_CSF.assign(p_CSF_new)
    else:
        p_CSF_new = float(p_CSF) + dt/Compliance(float(p_CSF))*(CSF_prod*1e-9 - 1/R_AG*(float(p_CSF) - p_AG) - 1/R_crib*(float(p_CSF) - p_crib))
        p_CSF.assign(p_CSF_new)
        
    os.system(f'echo "\nP_CSF = {float(p_CSF)/133.322} mmHg" >> log_{patient}.txt')
    [bc.apply(b) for bc in bcu]
    [bc.apply(A) for bc in bcu]
    solve(A, p_.vector(), b, 'gmres', 'hypre_euclid')
    
    p_mmHg.vector()[:] = p_.vector()/133.322

    
    k += 1
    #Skriver til fil
    if k % 10 == 0:
        tag += 1
        sol_pa.write_checkpoint(p_mmHg.sub(0), "p", tag,  XDMFFile.Encoding.HDF5, True)
        sol_pc.write_checkpoint(p_mmHg.sub(1), "p", tag,  XDMFFile.Encoding.HDF5, True)
        sol_pv.write_checkpoint(p_mmHg.sub(2), "p", tag,  XDMFFile.Encoding.HDF5, True)
        sol_ppa.write_checkpoint(p_mmHg.sub(3), "p", tag, XDMFFile.Encoding.HDF5, True)
        sol_ppc.write_checkpoint(p_mmHg.sub(4), "p", tag, XDMFFile.Encoding.HDF5, True)
        sol_ppv.write_checkpoint(p_mmHg.sub(5), "p", tag, XDMFFile.Encoding.HDF5, True)
        sol_pe.write_checkpoint(p_mmHg.sub(6), "p", tag,  XDMFFile.Encoding.HDF5, True)
    elif k ==1:
        sol_pa.write_checkpoint(p_mmHg.sub(0), "p", tag,  XDMFFile.Encoding.HDF5, False)
        sol_pc.write_checkpoint(p_mmHg.sub(1), "p", tag,  XDMFFile.Encoding.HDF5, False)
        sol_pv.write_checkpoint(p_mmHg.sub(2), "p", tag,  XDMFFile.Encoding.HDF5, False)
        sol_ppa.write_checkpoint(p_mmHg.sub(3), "p", tag, XDMFFile.Encoding.HDF5, False)
        sol_ppc.write_checkpoint(p_mmHg.sub(4), "p", tag, XDMFFile.Encoding.HDF5, False)
        sol_ppv.write_checkpoint(p_mmHg.sub(5), "p", tag, XDMFFile.Encoding.HDF5, False)
        sol_pe.write_checkpoint(p_mmHg.sub(6), "p", tag,  XDMFFile.Encoding.HDF5, False)

    # Beregner gjennomsnittsverdier. 
    pa_avg    = (1/133.322)*assemble(p_.sub(0)*dx(mesh))/Volume
    pc_avg    = (1/133.322)*assemble(p_.sub(1)*dx(mesh))/Volume
    pv_avg    = (1/133.322)*assemble(p_.sub(2)*dx(mesh))/Volume
    p_pa_avg  = (1/133.322)*assemble(p_.sub(3)*dx(mesh))/Volume
    p_pc_avg  = (1/133.322)*assemble(p_.sub(4)*dx(mesh))/Volume
    p_pv_avg  = (1/133.322)*assemble(p_.sub(5)*dx(mesh))/Volume
    p_ISF_avg = (1/133.322)*assemble(p_.sub(6)*dx(mesh))/Volume
    if int(mpi_comm.rank) == 0:
        pressure_averages.write("%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n"%(t, pa_avg,pc_avg, pv_avg, p_pa_avg,p_pc_avg,p_pv_avg,p_ISF_avg))
    t += dt
    
    os.system(f'echo "t = {t}" >> log_{patient}.txt')
    os.system(f'echo "t  = {t} s" >> results/avgs.txt')
    os.system(f'echo "P_a = {pa_avg} mmHg" >> results/avgs.txt')
    os.system(f'echo "P_c = {pc_avg} mmHg" >> results/avgs.txt')
    os.system(f'echo "P_v = {pv_avg} mmHg" >> results/avgs.txt')
    os.system(f'echo "P_aPVS = {p_pa_avg} mmHg" >> results/avgs.txt')
    os.system(f'echo "P_cPVS = {p_pc_avg} mmHg" >> results/avgs.txt')
    os.system(f'echo "P_vPVS = {p_pv_avg} mmHg" >> results/avgs.txt')
    os.system(f'echo "P_ISF =  {p_ISF_avg} mmHg" >> results/avgs.txt')

    #Beregner gjennomsnittsverdier per subdomain

      #Beregner gjennomsnittsverdier per subdomain
    p1_ = p_.split(True)
    for i, region in enumerate(brain_regions):
        if region_volumes[i] != 0:

            for j in range(7):
                avg_p = assemble(Constant(1)*p1_[j]*dxs[i])/region_volumes[i]/133.33

                P[j] = avg_p
        if int(mpi_comm.rank) == 0:
            out[i].write('%g %g %g %g %g %g %g %g \n'%(t, P[0], P[1], P[2], P[3], P[4], P[5], P[6]))
    p_1.assign(p_)
sol_pa.close() 
sol_pc.close() 
sol_pv.close() 
sol_pe.close() 
sol_ppa.close()
sol_ppc.close()
sol_ppv.close()
if int(mpi_comm.rank) == 0:
    [out[i].close() for i in range(len(brain_regions))]
    pressure_averages.close()

end = time.time()
os.system(f'echo "Simulation time = {end-start}" >> log_{patient}.txt')
