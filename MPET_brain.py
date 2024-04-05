
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--order", type=int, default=1)
parser.add_argument("--variation", type=str, default="base")
parser.add_argument("-res", "--resolution", type=int, default=16)
parser.add_argument("--xdmf", type=int, default=0)
args = parser.parse_args()
case = args.variation

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

T = 4200
dt = 20
patient = "C0"
res = args.resolution

order = args.order
xdmf = args.xdmf
main_dir = f"results/{patient}/"
if case.lower() == "base":
    out_dir_names = [main_dir + case.lower() + "/"]
elif case.lower() == "var1":
    out_dir_names = [f"{main_dir}/base/", f"{main_dir}/Res_FaghihSharp/", f"{main_dir}/Res_Pizzo/"]
elif case.lower() == "var2":
    out_dir_names = [f"{main_dir}/base/", f"{main_dir}/high_DS_resistance/", f"{main_dir}/low_DS_resistance/"]
elif case.lower() == "var3":
    out_dir_names = [f"r{main_dir}/constant_filtration/", f"{main_dir}/nonconstant_filtration/"]
elif case.lower() == "var4":
    out_dir_names = [f"{main_dir}/transfer_case{i}" for i in range(num_runs)]
    out_dir_names[0] = "base"
elif case.lower() == "var5":
    out_dir_names = [f"results/{patient}/base/", f"results/{patient}/high_kECS/", f"results/{patient}/very_high_kECS/"]

if not os.path.isdir("results/"):
    os.system("mkdir results")
    os.system(f"mkdir {main_dir}")

parameters["krylov_solver"]["maximum_iterations"] = 5000
parameters["krylov_solver"]["monitor_convergence"] = True
#parameters["krylov_solver"]["relative_tolerance"] = 2e-5

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

#List of measures to circumvent FEniCS integration bug
dxs = [dx(domain=mesh, subdomain_data=SD, subdomain_id=i) for i in range(1,4)]

mpi_comm = MPI.comm_world

geo = mesh.ufl_cell()
h = mesh.hmin()

compartments = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
Nc = len(compartments)

brain_regions = [1,2,3]

P1 = FiniteElement('CG',geo,order)
ME = MixedElement(Nc*[P1])
V = FunctionSpace(mesh,ME)
p = TrialFunctions(V)
q = TestFunctions(V)

R = FunctionSpace(mesh, P1)

CBVf = 0.033 #Barbacaru
porosity = [0.33*CBVf, 0.1*CBVf, 0.57*CBVf]
porosity.append(1.4*porosity[0])
porosity.append(1*porosity[1])
porosity.append(1.4*porosity[2])
porosity.append(0.14)

connections = np.array([[0,1,0,1,0,0,0],
                       [1,0,1,0,1,0,0],
                       [0,1,0,0,0,1,0],
                       [1,0,0,0,1,0,1],
                       [0,1,0,1,0,1,1],
                       [0,0,1,0,1,0,1],
                       [0,0,0,1,1,1,0]])

# PHYSICAL PARAMETERS
# FLUID
rho_f = Constant(1./1000)		# g/(mm^3)
nu_f = Constant(0.75)			# mm**2/s
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


r_pc = np.ones(num_runs)*9.2e-3*r_factor
if case.lower() == "var1":
    r_pc[1] = 32.24*r_factor
    r_pc[2] = 3.32e-4*r_factor

white_area = assemble(interpolate(Constant(1), R)*ds(2))



conductivity_ECS = 20*1e-18/mu_f          # m^2/(Pa*s)
constant = r_ECS*conductivity_ECS/mu_f # [1/m]

k_a = 1e6*constant*mu_b/(r_a*r_factor)     # constant is SI, mu_b is SI, r_a is SI thus k_a is SI [m^2s/Pa]
k_c = 1e6*1.44e-15/mu_b
k_v = 1e6*constant*mu_b/(r_v*r_factor)
k_pa = 1e6*constant*mu_f/r_pa
k_pc = 1e6*constant*mu_f/r_pc
k_pv = 1e6*constant*mu_f/r_pv
k_ECS = np.zeros(num_runs)
   

if case.lower() == "var5":
    k_ECS[0] = 1e6*conductivity_ECS
    k_ECS[1] = 1e7*conductivity_ECS
    k_ECS[2] = 1e8*conductivity_ECS
else:
    for i in range(num_runs):
        k_ECS[i] = 1e6*conductivity_ECS
n = FacetNormal(mesh)

t = 0

beta1 = 1e-3 # [1/(Pa*s)]  periarterial inflow
beta2 = 1e-3   # Venous outflow
beta3 = 1e-7   # perivenous outflow


CSF_prod = 0.33*1e3/60.
CSF_infusion = 1.5*1e3/60.
CSF_in = CSF_prod+CSF_infusion

ventricle_area = assemble(interpolate(Constant(1), R)*ds(2))
surface_area = assemble(interpolate(Constant(1), R)*(ds(1)+ ds(2)+ds(3)))
pia_area = assemble(interpolate(Constant(1), R)*ds(1))

Volume = assemble(1*dx(mesh))
Rout = 11.0098 - 1.5294  #Rout set s.t. alpha = 1
blood_flow = 712.5
p0 = 9

b_avg = blood_flow*1e3/60./pia_area # mL/min = cm^3/min = 1e3*mm^3/(60s)
b_in = Constant(b_avg)

w_a_c = b_avg*pia_area/(Volume*60*133.33)   # For a ~60 mmHg drop from average Arterial to Capillary Pressure
w_c_v = b_avg*pia_area/(Volume*12.5*133.33) # For a ~12.5 mmHg drop from average Capillary to Venous pressure
w_pc_c = np.ones(num_runs)*(Volume*1e-9*r_cv)**-1
w_pv_v = 1e-15 # flow into veins
w_pa_a = 1e-15 # flow from arteries to arterial PVS
w_pa_pc = 1e-6
w_pc_pv = 1e-6
w_pa_pv = 0
w_a_pv  = 0
w_a_pc  = 0
w_a_v   = 0
w_c_pa  = 0
w_c_pv  = 0
w_v_pa  = 0
w_v_pc = 0
if case.lower() == "var3":
    w_pc_c[0] = 0
    

w_e_a = 0
w_c_e = 0
w_e_v = 0
w_pv_e = (Volume*1e-9*r_IEG_v)**-1*np.ones(num_runs)    # Omega is 1/(V*R) ~ 1/(Pa*s/m^3*m^3) = 1/(Pa*s) 
w_pa_e = (Volume*1e-9*r_IEG_a)**-1*np.ones(num_runs)  

w_pc_e = 1e-10*np.ones(num_runs) # flow from capillary PVS to extracellular matrix 

if case.lower() == "var4":
    w_pc_e[1] = 5e-7
    w_pc_e[2] = 1e-8
    w_pv_e[2] = w_pv_e[0]*10
    w_pa_e[2] = w_pa_e[0]*10
    w_pc_e[3] = w_pa_e[0]
    w_pa_e[3] = w_pc_e[0]
    w_pa_e[4] = w_pa_e[0]/2
    w_pv_e[4] = w_pv_e[0]/2

def Compliance(p,E = 0.2e6, p0 = p0*133.33):
    if p < 13*13.322:
        if p0 > 13*133.322:
            p0 = 12*133.322
        return 1./(E*(13*133.322 - p0))
    else:
        if p0 > p:
            p0 = p - 2*133.322
        return 1./(E*(p-p0))

C_array = [1e-4, 1e-8, 1e-4, Constant(1e-8), Constant(1e-8), Constant(1e-8), 1e-8]
p_AG = 8.4*133.33           # AG pressure
p_CSF = Constant(10*133.33)  # CSF pressure
p_crib = Constant(0)        # cribriform plate pressure

for i_, var_dir in enumerate(out_dir_names):

    w = np.array([[0, w_a_c, w_a_v, w_pa_a, w_a_pc, w_a_pv, w_e_a],
              [w_a_c, 0, w_c_v, w_c_pa, w_pc_c[i_], w_c_pv, w_c_e], 
              [w_a_v, w_c_v, 0, w_v_pa, w_v_pc, w_pv_v, w_e_v],
              [w_pa_a, w_c_pa, w_v_pa, 0, w_pa_pc, w_pa_pv, w_pa_e[i_]],
              [w_a_pc, w_pc_c[i_], w_v_pc, w_pa_pc, 0, w_pc_pv, w_pc_e[i_]],
              [w_a_pv, w_c_pv, w_pv_v, w_pa_pv, w_pc_pv, 0, w_pv_e[i_]],
              [w_e_a, w_c_e, w_e_v, w_pa_e[i_], w_pc_e[i_], w_pv_e[i_], 0]])
    # Initial condition
    p_1 = interpolate(Expression(("80*133.322", "20*133.322", "10*133.322", "10.2*133.322", "9.2*133.322", "9.2*133.322", "9.2*133.322"), degree=1), V)

    # Bilinear form
    F = 0
    kappas = [Constant(k_a), Constant(k_c), Constant(k_v), Constant(k_pa), Constant(k_pc[i_]), Constant(k_pv), Constant(k_ECS[i_])]
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

    F += CSF_prod/ventricle_area*q[1]*ds(2) # Choroid plexus production. 

    F -= beta1*(p_CSF - p[3])*q[3]*ds(1)    # CSF-to-PVS-transfer modeled by ODE

    F -= beta2*((p_AG+p_CSF)/2 - p[2])*q[2]*ds(1)          # Venous outflow
    F -= beta3*((p_AG+p_CSF)/2 - p[5])*q[5]*ds(1)     # venous-PVS-to-lymph transfer, pressure = 8.4
    if case.lower() == "var3" and i_ == 0:
        filt_rate = 0.16
        cap_avg = filt_rate*1e3/60./white_area # mL/min = cm^3/min = 1e3*mm^3/(60s)
        C_in = Constant(cap_avg)
        F -= C_in*q[4]*(ds(2))
        F += C_in*q[1]*(ds(2))

    bcu = []#[bcp]

    # System matrix assembly
    A = assemble(lhs(F))
    p_ = Function(V)

    t = 0

    # Convert Rout to alpha
    alpha = (Rout + 1.5294)/11.0098

    R_DS = 10.81*r_factor*alpha*np.ones(num_runs)
    if case.lower() == "var2":
        R_DS[1] = R_DS[0]*2
        R_DS[2] = R_DS[0]/2
    
    R_crib = 67*r_factor*alpha

    if int(mpi_comm.rank) == 0:
        os.system(f"mkdir {var_dir}")
        pressure_averages = open(f"{var_dir}pressure_average.dat", 'w+')
        pressure_averages.write("Time Arterial Capillary Venous PVS_a PVS_c PVS_v ISF CSF\n")
        out_pressure = [open(f"{var_dir}"+'avg_pressure_%d.dat'%region,'w') for region in brain_regions]
        [out_pressure[i].write("Time Arterial Capillary Venous PVS_a PVS_c PVS_v ISF CSF\n") for i in range(len(brain_regions))]

        speed_averages = open(f"{var_dir}speed_average.dat", 'w+')
        speed_averages.write("Time Arterial Capillary Venous PVS_a PVS_c PVS_v ISF\n")    
        out_speed = [open(f"{var_dir}"+'avg_speed_%d.txt'%region,'w') for region in brain_regions]
        [out_speed[i].write("Time Arterial Capillary Venous PVS_a PVS_c PVS_v ISF\n") for i in range(len(brain_regions))]

    region_volumes = [assemble(1*dx(region, domain=mesh)) for region in brain_regions]

    P = np.zeros(8)
    Varray = np.zeros(7)

    p_mmHg = Function(V)
    p_mmHg.vector()[:] = p_1.vector()/133.322
    k = 0

    if xdmf:
        os.system(f"mkdir {var_dir}pvds")
        sol_pa = XDMFFile(MPI.comm_world,  f"{var_dir}pvds/p_a.xdmf")
        sol_pc = XDMFFile(MPI.comm_world,  f"{var_dir}pvds/p_c.xdmf")
        sol_pv = XDMFFile(MPI.comm_world,  f"{var_dir}pvds/p_v.xdmf")
        sol_pe = XDMFFile(MPI.comm_world,  f"{var_dir}pvds/p_e.xdmf")
        sol_ppa = XDMFFile(MPI.comm_world, f"{var_dir}pvds/p_pa.xdmf")
        sol_ppc = XDMFFile(MPI.comm_world, f"{var_dir}pvds/p_pc.xdmf")
        sol_ppv = XDMFFile(MPI.comm_world, f"{var_dir}pvds/p_pv.xdmf")


    tag = 0

    ISF_2_PVSa  = 0
    PVSA_2_PVSc = 0
    PVSA_2_A    = 0

    # Integration loop
    while t < T:
        b = assemble(rhs(F))
        if abs(1000 - t) <= 100 and abs(1000 - t) > 20:
            dt = 10
        elif abs(1000 - t) < 20:
            dt = 2
        else:
            dt = 20

        # ODE for new CSF pressure on boundary.
        if t > 1000 and t < 3000:
            # Infusion
            p_CSF_new = float(p_CSF) + dt/Compliance(float(p_CSF))*((CSF_in - 1000/60 * abs(PVSA_2_PVSc + ISF_2_PVSa + PVSA_2_A))*1e-9 - 1/R_DS[i_]*(float(p_CSF) - p_AG) - 1/R_crib*(float(p_CSF) - p_crib))
            p_CSF.assign(p_CSF_new)
        else:
            # Not infusion
            p_CSF_new = float(p_CSF) + dt/Compliance(float(p_CSF))*((CSF_prod- 1000/60 * abs(PVSA_2_PVSc + ISF_2_PVSa + PVSA_2_A))*1e-9 - 1/R_DS[i_]*(float(p_CSF) - p_AG) - 1/R_crib*(float(p_CSF) - p_crib))
            p_CSF.assign(p_CSF_new)

        solve(A, p_.vector(), b, 'gmres', 'hypre_euclid')

        p_mmHg.vector()[:] = p_.vector()/133.322


        k += 1
        # Write to XDMF file
        if xdmf:
          if k % 45 == 0:
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

        # Compute average values . 
        pa_avg    = (1/133.322)*assemble(p_.sub(0)*dx(mesh))/Volume
        pc_avg    = (1/133.322)*assemble(p_.sub(1)*dx(mesh))/Volume
        pv_avg    = (1/133.322)*assemble(p_.sub(2)*dx(mesh))/Volume
        p_pa_avg  = (1/133.322)*assemble(p_.sub(3)*dx(mesh))/Volume
        p_pc_avg  = (1/133.322)*assemble(p_.sub(4)*dx(mesh))/Volume
        p_pv_avg  = (1/133.322)*assemble(p_.sub(5)*dx(mesh))/Volume
        p_ISF_avg = (1/133.322)*assemble(p_.sub(6)*dx(mesh))/Volume
        va_avg    = 1e6*(kappas[0]/(porosity[0]))*assemble(inner(grad(p_.sub(0)), grad(p_.sub(0)))**(0.5)*dx(mesh))/Volume
        vc_avg    = 1e6*(kappas[1]/(porosity[1]))*assemble(inner(grad(p_.sub(1)), grad(p_.sub(1)))**(0.5)*dx(mesh))/Volume
        vv_avg    = 1e6*(kappas[2]/(porosity[2]))*assemble(inner(grad(p_.sub(2)), grad(p_.sub(2)))**(0.5)*dx(mesh))/Volume
        v_pa_avg  = 1e6*(kappas[3]/(porosity[3]))*assemble(inner(grad(p_.sub(3)), grad(p_.sub(3)))**(0.5)*dx(mesh))/Volume
        v_pc_avg  = 1e6*(kappas[4]/(porosity[4]))*assemble(inner(grad(p_.sub(4)), grad(p_.sub(4)))**(0.5)*dx(mesh))/Volume
        v_pv_avg  = 1e6*(kappas[5]/(porosity[5]))*assemble(inner(grad(p_.sub(5)), grad(p_.sub(5)))**(0.5)*dx(mesh))/Volume
        v_ISF_avg = 1e6*(kappas[6]/(porosity[6]))*assemble(inner(grad(p_.sub(6)), grad(p_.sub(6)))**(0.5)*dx(mesh))/Volume


        ISF_2_PVSa = 60/1000 *assemble(Constant(1)*w[3,6]*(p_.sub(3) - p_.sub(6))*dx(mesh)) 
        PVSA_2_PVSc = 60/1000 *assemble(Constant(1)*w[3,4]*(p_.sub(3) - p_.sub(4))*dx(mesh))
        PVSA_2_A = 60/1000 *assemble(Constant(1)*w[0,3]*(p_.sub(3) - p_.sub(0))*dx(mesh))
        for i, region in enumerate(brain_regions):
            if region_volumes[i] != 0:
                for j in range(7):
                    avg_p = assemble(Constant(1)*p_.sub(j)*dxs[i])/region_volumes[i]/133.322
                    avg_v = (kappas[j]/(porosity[j]*133.322))*assemble(inner(grad(p_.sub(j)), grad(p_.sub(j)))**(0.5)*dxs[i])/region_volumes[i]
                    P[j] = avg_p
                    Varray[j] = avg_v
                P[-1] = p_CSF
                if int(mpi_comm.rank) == 0:
                    out_pressure[i].write('%g %g %g %g %g %g %g %g %g \n'%(t, P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7]))
                    out_speed[i].write('%g %g %g %g %g %g %g %g \n'%(t, Varray[0], Varray[1], Varray[2], Varray[3], Varray[4], Varray[5], Varray[6]))
        #Transfer data
        for i in range(7):
            connected = connections[i,:]
            for j in range(7):
                if connected[j]>0:
                    transfer_parenchyma =  60/1000 *assemble(Constant(1)*w[i,j]*(p_.sub(i) - p_.sub(j))*dx(mesh)) # (Pas)^-1 * Pa * mm^3 = mm^3/s = 1e-9 m^3/s = 1e-6 l/s 
                    transfer_grey      =  60/1000 *assemble(Constant(1)*w[i,j]*(p_.sub(i) - p_.sub(j))*dx(1,domain=mesh))
                    transfer_white      =  60/1000 *assemble(Constant(1)*w[i,j]*(p_.sub(i) - p_.sub(j))*dx(2,domain=mesh))
                    if int(mpi_comm.rank) == 0: 
                        if k == 1:
                            os.system(f'echo "Time Parenchyma Grey White" >> {var_dir}transfer_{compartments[i]}_{compartments[j]}.txt')
                        os.system(f'echo "{t} {transfer_parenchyma} {transfer_grey} {transfer_white}" >> {var_dir}transfer_{compartments[i]}_{compartments[j]}.txt')


        if int(mpi_comm.rank) == 0:
            pressure_averages.write("%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n"%(t, pa_avg,pc_avg, pv_avg, p_pa_avg,p_pc_avg,p_pv_avg,p_ISF_avg, p_CSF/133.322))

            speed_averages.write("%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n"%(t, va_avg,vc_avg, vv_avg, v_pa_avg,v_pc_avg,v_pv_avg,v_ISF_avg))

        t += dt
        p_1.assign(p_)
    if xdmf:
      sol_pa.close() 
      sol_pc.close() 
      sol_pv.close() 
      sol_pe.close() 
      sol_ppa.close()
      sol_ppc.close()
      sol_ppv.close()

    if int(mpi_comm.rank) == 0:
        [out_pressure[i].close() for i in range(len(brain_regions))]
        [out_speed[i].close() for i in range(len(brain_regions))]
        speed_averages.close()
        pressure_averages.close()
