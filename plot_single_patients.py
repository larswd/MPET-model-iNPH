from pyclbr import Function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from IPython import embed


def read_xdmf(infile, mesh, order, tag, patient, comp):
    
    V = FunctionSpace(mesh, "CG", order)
    p = Function(V)
    try:
        p_xdmf = XDMFFile(MPI.comm_world, infile)
        p_xdmf.read_checkpoint(p, "p", tag)
        p_xdmf.close()
    except:
        if int(mpi_comm.rank) == 0:
            os.system(f"scp larswd@saga.sigma2.no:/cluster/projects/nn8017k/Lars/experiments/meshes/{patient}/results32/pvds/* patients/{patient}/")
        p_xdmf = XDMFFile(MPI.comm_world, infile)
        p_xdmf.read_checkpoint(p, "p", tag)
        p_xdmf.close()
    return p


def analyse_patient(patient, recompute_transfer=True, recompute_velocities=True, recompute_pressure_avg=True):
    
    mesh = Mesh()
    print(patient)
    comps = ["a", "c", "v", "pa", "pc", "pv", "e"]
    compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
    Nc = len(comps)
    try:
        hdf = HDF5File(mesh.mpi_comm(), f"patients/{patient}/parenchyma32.h5", "r")
        hdf.read(mesh, "/mesh", False)
    except:
        if int(mpi_comm.rank) == 0:
            os.system(f"scp larswd@saga.sigma2.no:/cluster/projects/nn8017k/Lars/experiments/meshes/{patient}/parenchyma32.h5 patients/{patient}/")
        hdf = HDF5File(mesh.mpi_comm(), f"patients/{patient}/parenchyma32.h5", "r")
        hdf.read(mesh, "/mesh", False)
    SD = MeshFunction("size_t", mesh,mesh.topology().dim())
    hdf.read(SD, "/subdomains")
    lut = MeshFunction("size_t", mesh,mesh.topology().dim())
    hdf.read(lut, "/lookup_table")
    bnd = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
    hdf.read(bnd, "/boundaries")
    ds = Measure('ds')(subdomain_data=bnd)
    dS = Measure('dS')(subdomain_data=bnd)
    dx = Measure('dx')(subdomain_data=SD)
    dxl = Measure('dx')(subdomain_data=lut)


    geo = mesh.ufl_cell()
    P1 = FiniteElement('CG',geo,1)
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
    #[3.63e-8, 1.44e-9, 1.13e-6, 2e-11]
    n = FacetNormal(mesh)

    Qprod = Expression("5.787e-5", degree=0)

    t = 0
    ventricle_area = assemble(interpolate(Constant(1), R)*ds(2))
    surface_area = assemble(interpolate(Constant(1), R)*(ds(1) + ds(2) + ds(3))) 
    pia_area = assemble(interpolate(Constant(1), R)*ds(1))
    Volume = assemble(1*dx(mesh))
    Volume_grey = assemble(1*dx(1,domain=mesh))
    Volume_white = assemble(1*dx(2,domain=mesh))
    Volume_amygdala = assemble(1*dxl(18,domain=mesh))
    Volume_hippocampus = assemble(1*dxl(53,domain=mesh))
    Volume_cortex = assemble(1*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))

    print(Volume)
    print(Volume_grey)
    print(Volume_white)    
    print(Volume_cortex)    
    print(Volume_hippocampus)    
    print(Volume_amygdala)    
    p_AG = 8.4*133.322
    if patient[0] == "C":
        blood_flow = 712.5
    else:
        blood_flow = 653.5
    b_avg = blood_flow*1e3/60./surface_area # mL/min = cm^3/min = 1e3*mm^3/(60s)
    b_in = Constant(b_avg)

    w_a_c = b_avg*surface_area/(Volume*60*133.33)   # For a ~60 mmHg drop from average Arterial to Capillary Pressure
    w_c_v = b_avg*surface_area/(Volume*12.5*133.33) # For a ~12.5 mmHg drop from average Capillary to Venous pressure


    w_pc_c = 0.1*(Volume*1e-9*r_cv)**-1
    w_pv_v = 1e-17 # flow into veins?   Justert opp. Muligens for stor
    w_pa_a = 1e-17 # flow from arteries to arterial PVS? Justert opp. Muligens for stor
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

    print(w_pa_e, w_pv_e)
    w_pc_e = 1e-15 # flow from capillary PVS to extracellular matrix ?


    w = np.array([[0, w_a_c, w_a_v, w_pa_a, w_a_pc, w_a_pv, w_e_a],
                  [w_a_c, 0, w_c_v, w_c_pa, w_pc_c, w_c_pv, w_c_e], 
                  [w_a_v, w_c_v, 0, w_v_pa, w_v_pc, w_pv_v, w_e_v],
                  [w_pa_a, w_c_pa, w_v_pa, 0, w_pa_pc, w_pa_pv, w_pa_e],
                  [w_a_pc, w_pc_c, w_v_pc, w_pa_pc, 0, w_pc_pv, w_pc_e],
                  [w_a_pv, w_c_pv, w_pv_v, w_pa_pv, w_pc_pv, 0, w_pv_e],
                  [w_e_a, w_c_e, w_e_v, w_pa_e, w_pc_e, w_pv_e, 0]])
    
    # Big chunks
    transfer_matrix_parenchyma = np.zeros((12,7,7))
    transfer_matrix_grey = transfer_matrix_parenchyma.copy()
    transfer_matrix_white = transfer_matrix_parenchyma.copy()

    # Small transfer areas
    transfer_matrix_amygdala = transfer_matrix_white.copy()
    transfer_matrix_cortex = transfer_matrix_parenchyma.copy()
    transfer_matrix_hippocampus = transfer_matrix_white.copy()
    
    ventricular_pressure_force = np.zeros((12,7))
    avg_vel             = np.zeros((12,7))
    avg_vel_grey        = np.zeros((12,7))
    avg_vel_white       = np.zeros((12,7))
    avg_vel_amygdala    = np.zeros((12,7))
    avg_vel_cortex      = np.zeros((12,7))
    avg_vel_hippocampus = np.zeros((12,7))

    avg_pressure             = np.zeros((12,7))
    avg_pressure_grey        = np.zeros((12,7))
    avg_pressure_white       = np.zeros((12,7))
    avg_pressure_amygdala    = np.zeros((12,7))
    avg_pressure_cortex      = np.zeros((12,7))
    avg_pressure_hippocampus = np.zeros((12,7))
    flux_out = np.zeros((12,7))
    t = np.zeros(12)
    # i compartmental index, k time index
    folder = f"C25_extended"
    print("Starting plotting")
    for i, comp in enumerate(comps):
        for k in range(12):
            infile = folder + f"/p_{comp}.xdmf"
            p = read_xdmf(infile, mesh, 2, k, patient, comp)
            print("File read")
            # Computing transfer between compartments
            for j, comp_ in enumerate(comps[i:]):
                if w[i,j] > 0:
                    p2 = read_xdmf(folder + f"/p_{comp_}.xdmf", mesh, 2, k, patient, comp)
                    transfer_matrix_parenchyma[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2 - p)*dx(mesh)) # (Pas)^-1 * Pa * mm^3 = mm^3/s = 1e-9 m^3/s = 1e-6 l/s 
                    transfer_matrix_grey[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2 - p)*dx(1,domain=mesh))
                    transfer_matrix_white[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2 - p)*dx(2,domain=mesh))

                    transfer_matrix_cortex[k,i,j] = 60/1000 * assemble(133.322*Constant(1)*w[i,j]*(p2 - p)*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))
                    transfer_matrix_hippocampus[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2-p)*dxl(53,domain=mesh))
                    transfer_matrix_amygdala[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2-p)*dxl(18,domain=mesh))
                    
                    transfer_matrix_cortex[k,i,j]      = -transfer_matrix_cortex[k,j,i]
                    transfer_matrix_grey[k,i,j]        = -transfer_matrix_grey[k,j,i]
                    transfer_matrix_hippocampus[k,i,j] = -transfer_matrix_hippocampus[k,j,i]
                    transfer_matrix_parenchyma[k,i,j]  = -transfer_matrix_parenchyma[k,j,i]
                    transfer_matrix_amygdala[k,i,j]    = -transfer_matrix_amygdala[k,j,i] 
                    transfer_matrix_white[k,i,j]       = -transfer_matrix_white[k,j,i]
                    print("\n")
                    print(k, comps[i], comps[j])
                    print(w[i,j])
                    print(transfer_matrix_cortex[k,i,j])
                    print(transfer_matrix_grey[k,i,j])
                    print(transfer_matrix_hippocampus[k,i,j])
                    print(transfer_matrix_parenchyma[k,i,j])
                    print(transfer_matrix_amygdala[k,i,j])
                    print(transfer_matrix_white[k,i,j])
            
            avg_vel[k,i]             = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p), grad(p))**(0.5)*dx(mesh))/Volume #Units: grad(p) Pa/mm, kappa m^2/(sPa) dx: mm^3 - unit m^2/mm s ~ km/s
            avg_vel_grey[k,i]        = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p), grad(p))**(0.5)*dx(1,domain=mesh))/Volume_grey
            avg_vel_white[k,i]       = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p), grad(p))**(0.5)*dx(2,domain=mesh))/Volume_white
            avg_vel_amygdala[k,i]    = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p), grad(p))**(0.5)*dxl(18,domain=mesh))/Volume_amygdala
            avg_vel_cortex[k,i]      = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p), grad(p))**(0.5)*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))/Volume_cortex
            avg_vel_hippocampus[k,i] = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p), grad(p))**(0.5)*dxl(53,domain=mesh))/Volume_hippocampus

            print(avg_vel[k,i])
            print(avg_vel_cortex[k,i])
            print(avg_vel_grey[k,i])
            print(avg_vel_hippocampus[k,i])
            print(avg_vel_amygdala[k,i])
            print(avg_vel_white[k,i])

            avg_pressure[k,i]             = assemble(p*Constant(1)*dx(mesh))/Volume
            avg_pressure_grey[k,i]        = assemble(p*Constant(1)*dx(1,domain=mesh))/Volume_grey
            avg_pressure_white[k,i]       = assemble(p*Constant(1)*dx(2,domain=mesh))/Volume_white
            avg_pressure_amygdala[k,i]    = assemble(p*Constant(1)*dxl(18,domain=mesh))/Volume_amygdala
            avg_pressure_cortex[k,i]      = assemble(p*Constant(1)*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))/Volume_cortex
            avg_pressure_hippocampus[k,i] = assemble(p*Constant(1)*dxl(53,domain=mesh))/Volume_hippocampus

            print(avg_pressure[k,i])
            print(avg_pressure_cortex[k,i])
            print(avg_pressure_grey[k,i])
            print(avg_pressure_hippocampus[k,i])
            print(avg_pressure_amygdala[k,i])
            print(avg_pressure_white[k,i])

            ventricular_pressure_force[k,i] = 1e-6*assemble(133.322*Constant(1)*p*ds(2)) # N/m^2 * mm^2 = 1e-6 N
            flux_out[k,i] = 60*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p), n)*ds(1)) #m/s * mm^2 = 1e-6 m^3/s = 1e-3 l/s = ml/s

            print(ventricular_pressure_force[k,i])
            print(flux_out[k,i])
            t[k] = k*200
            print(transfer_matrix_parenchyma[k,i,:])

        try:
            os.system("mkdir " +folder + "/plots")
        except:
            pass

    return transfer_matrix_parenchyma, transfer_matrix_grey, transfer_matrix_white, transfer_matrix_cortex, \
           transfer_matrix_hippocampus, transfer_matrix_amygdala, avg_pressure, avg_pressure_grey, \
           avg_pressure_white, avg_pressure_cortex, avg_pressure_amygdala, avg_pressure_hippocampus, \
           avg_pressure_amygdala, avg_vel, avg_vel_grey, avg_vel_white, avg_vel_cortex, avg_vel_hippocampus, \
           avg_vel_amygdala, ventricular_pressure_force, t, w



regions = ["parenchyma", "grey matter", "white matter", "part of cortex", "hippocampus", "amygdala"]
comps = ["a", "c", "v", "pa", "pc", "pv", "e"]
compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]

transfer_matrix_parenchyma, transfer_matrix_grey, transfer_matrix_white, transfer_matrix_cortex, \
transfer_matrix_hippocampus, transfer_matrix_amygdala, avg_pressure, avg_pressure_grey, \
avg_pressure_white, avg_pressure_cortex, avg_pressure_amygdala, avg_pressure_hippocampus, \
avg_pressure_amygdala, avg_vel, avg_vel_grey, avg_vel_white, avg_vel_cortex, avg_vel_hippocampus, \
avg_vel_amygdala, ventricular_pressure_force, t, w = analyse_patient("C25")

group = "C"

transfer_data = [transfer_matrix_parenchyma, transfer_matrix_grey, transfer_matrix_white, transfer_matrix_cortex, \
    transfer_matrix_hippocampus, transfer_matrix_amygdala]
other_data =  [avg_pressure, avg_pressure_grey, avg_pressure_white, avg_pressure_cortex, avg_pressure_hippocampus, \
    avg_pressure_amygdala, avg_vel, avg_vel_grey, avg_vel_white, avg_vel_cortex, avg_vel_hippocampus, avg_vel_amygdala]
m = 0
for k, data in enumerate(transfer_data):    
    for i in range(7):
        for j in range(i,7):
            if abs(w[i,j]) > 0:
                plt.plot(t, data[:,i,j])
                plt.xlabel("Time [s]")
                plt.ylabel("Volume flux [(ml/min)/mm$^{-3}$]")
                plt.title(f"Total transfer in {regions[k]}")
                plt.legend(["C25"])
                outfile = "C25_extended" + f"/plots/transfer_{regions[k]}_{compartment_names[i]}_{compartment_names[j]}_{group}"
                plt.savefig(outfile + ".pdf")
                plt.close()
                with open(outfile + ".txt", 'w+') as write_file:
                    write_file.write(f"t {t}\n")
                    write_file.write(f"transfer_avg {data[:,i,j]}\n")
for k, data in enumerate(other_data):
    for i in range(7):            
        plt.plot(t, data[:,i])
        plt.xlabel("Time [s]")
        if m > 6:
            plt.ylabel("Speed [$\mu$m/s]")      
            plt.title(f"Average velocity in {regions[k%6]}")
            outfile = "C25_extended" + f"/plots/avg_vel_{regions[k%6]}_{compartment_names[i]}_{group}"
        else:
            plt.ylabel("Pressure [mmHg]")      
            plt.title(f"Average pressure in {regions[k%6]}")
            outfile = "C25_extended" + f"/plots/avg_pressure_{regions[k%6]}_{compartment_names[i]}_{group}"
        plt.legend(["Mean"])
        plt.savefig(outfile + ".pdf")
        plt.close()
        with open(outfile + ".txt", 'w+') as write_file:
            write_file.write(f"t {t}\n")
            write_file.write(f"other_avg {other_data[k][:,i]}\n")
        m += 1