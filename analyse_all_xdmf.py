from pyclbr import Function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from IPython import embed


mpi_comm = MPI.comm_world

def read_xdmf(infile, mesh, order, tag, patient, comp):
    
    V = FunctionSpace(mesh, "CG", order)
    p = Function(V)
    p_xdmf = XDMFFile(MPI.comm_world, infile)
    p_xdmf.read_checkpoint(p, "p", tag)
    p_xdmf.close()
    #except:
    #    if int(mpi_comm.rank) == 0:
    #        os.system(f"scp larswd@saga.sigma2.no:/cluster/projects/nn8017k/Lars/experiments/meshes/{patient}/results32/pvds/* patients/{patient}/")
    #    p_xdmf = XDMFFile(MPI.comm_world, infile)
    #    p_xdmf.read_checkpoint(p, "p", tag)
    #    p_xdmf.close()
    return p

def clean_data(infile, outfile):

    with open(infile, 'r') as infile:
        lines = infile.readlines()
        j = 0
        for i,line in enumerate(lines):
            completed = False
            k = i
            while not completed:   
                if k + 1 < len(lines):
                    if lines[i+1][0] == 'o' or lines[i+1][0] == 't':
                        completed = True

                        if lines[i][-2:] == "]\n":
                            if lines[i][-3] == " ":
                                lines[i] = lines[i][:-3] + lines[i][-2:]
                                completed = False
                        elif lines[i][-1] == "\n" and (lines[i][-2:] != "]\n" and lines[i][-2:] != " \n"):
                            lines[i] = lines[i][:-1]
                            lines[i] += "]\n"
                        elif lines[i][-1] == "\n" and lines[i][-2] == " ":
                            lines[i] = lines[i][:-2] + "\n"
                            completed = False
                        else:
                            lines[i] += "]\n"

                        

                    else:
                        lines_tmp = lines[:(i+1)]
                        lines_tmp[-1] = lines_tmp[-1][:-1]
                        lines_tmp[-1] += lines[i+1]
                        for line in lines[(i+2):]:
                            lines_tmp.append(line)
                        lines = lines_tmp
                else:
                    completed = True
                k += 1
    
    if lines[-1][0] != "o" and lines[-1][0] != "t":
        lines[-2] = lines[-2][:-1] + lines[-1] + "\n"
        lines = lines[:-1]

    last_line_white_space = True
    while last_line_white_space:
        last_line_white_space = False
        if lines[-1][-2:] == "]\n":
            if lines[-1][-3] == " ":
                lines[-1] = lines[-1][:-3] + lines[-1][-2:]
                completed = True
            pass
        elif lines[-1][-1] == "\n" and (lines[-1][-2:] != "]\n" and lines[-1][-2:] != " \n"):

            lines[-1] = lines[-1][:-1]
            lines[-1] += "]\n"
        elif lines[-1][-1] == "\n" and lines[-1][-2] == " ":

            lines[-1] = lines[-1][:-2] + "\n"
            last_line_white_space = True
        elif lines[-1][-2:] == " ]":
            lines[-1] = lines[-1][:-2] + lines[-1][-1]
            last_line_white_space = True
        else:
            lines[-1] += "]\n"
    
    if lines[-1][-3:] == "]]\n":
        lines[-1] = lines[-1][:-2] + "\n"
    with open(outfile, 'w') as outfile:
        for line in lines:
            line2 = line.split()
            if line2[1] == "[":
                line2_tmp = line2[:2]
                line2_tmp[1] += line2[2]
                for i in range(len(line2[3:])):
                    line2_tmp.append(line2[3+i])
                line_tmp = ""
                for s in line2_tmp:
                    line_tmp += s + " "
                line = line_tmp
            if line[-1] != "\n":
                line += "\n"
            outfile.write(line)



def analyse_patient(patient, recompute_transfer=True, recompute_velocities=True, recompute_pressure_avg=True):
    
    mesh = Mesh()
    print(patient)
    comps = ["a", "c", "v", "pa", "pc", "pv", "e"]
    compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
    regions = ["parenchyma", "grey_matter", "white_matter", "cingulum", "hippocampus", "amygdala"]
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

    print("Test 1")
    geo = mesh.ufl_cell()
    P1 = FiniteElement('CG',geo,1)
    R = FunctionSpace(mesh, P1)


    # PHYSICAL PARAMETERS (hentet fra Vegards skript)
    # FLUID
    rho_f = Constant(1./1000)		# g/(mm^3)
    nu_f = Constant(0.658)			# mm**2/s
    mu_f = Constant(nu_f*rho_f)		# g/(mm*s)

    mu_b = 3*mu_f

    #Funket hit
    
    r_a = 0.000939857688751   # mmHg/mL/min    these two based on our calculation of resistance from F&S for vessels (not PVS)
    r_v = 8.14915973766e-05   # mmHg/mL/min    script: /Waterscape/Waterscape/PhD/paperIV/Stoverud/compartment-ode-model 

    r_factor = 133*1e6*60     # mmHg/(mL/min) = 133 Pa/(1e-3 L / 60 s) = 133 Pa / (1e-3 * 1e-3 m^3/ 60s) = 133*1e6*60 Pa/s
    r_ECS = 0.57*r_factor     # Pa/(m^3/s)    (from ECS permeability of Karl-Erik together with the 2D geometry by Adams)
    print("Test 2")
    r_IEG_v = 0.64*r_factor    # venous inter-endfeet gaps
    r_IEG_a = 0.57*r_factor    # arterial inter-endfeet gaps
    r_cv = 125*r_factor         # resistance over capillary wall
    r_pa = 1.02*r_factor        # resistance along periarterial spaces
    r_pv = 0.079*r_factor       # resistance along perivenous spaces
    r_pc = 32.24*r_factor       # resistance along pericapillary spaces

    #Men kom ikke hit
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
    surface_area = assemble(interpolate(Constant(1), R)*(ds(1) + ds(2) + ds(3))) 
    Volume = assemble(1*dx(mesh))

    p_AG = 8.4*133.322
    try:
        with open(f"patients/{patient}/patdat.txt", 'r') as patfile:
            lines = patfile.readlines()
            blood_flow = float(lines[1].split()[1])
    except Exception:
        os.system(f"scp larswd@saga.sigma2.no:/cluster/projects/nn8017k/Lars/experiments/meshes/{patient}/patdat.txt patients/{patient}/patdat.txt")
        with open(f"patients/{patient}/patdat.txt", 'r') as patfile:
            lines = patfile.readlines()
            blood_flow = float(lines[1].split()[1])
            
    
    print("Blood_flow", blood_flow)
    b_avg = blood_flow*1e3/60./surface_area # mL/min = cm^3/min = 1e3*mm^3/(60s)
    b_in = Constant(b_avg)

    w_a_c = b_avg*surface_area/(Volume*60*133.33)   # For a ~60 mmHg drop from average Arterial to Capillary Pressure
    w_c_v = b_avg*surface_area/(Volume*12.5*133.33) # For a ~12.5 mmHg drop from average Capillary to Venous pressure


    w_pc_c = (Volume*1e-9*r_cv)**-1
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
    transfer_matrix_parenchyma = np.zeros((13,7,7))
    transfer_matrix_grey = transfer_matrix_parenchyma.copy()
    transfer_matrix_white = transfer_matrix_parenchyma.copy()

    # Small transfer areas
    transfer_matrix_amygdala = transfer_matrix_white.copy()
    transfer_matrix_cortex = transfer_matrix_parenchyma.copy()
    transfer_matrix_hippocampus = transfer_matrix_white.copy()
    
    ventricular_pressure_force = np.zeros((13,7))
    avg_vel             = np.zeros((13,7))
    avg_vel_grey        = np.zeros((13,7))
    avg_vel_white       = np.zeros((13,7))
    avg_vel_amygdala    = np.zeros((13,7))
    avg_vel_cortex      = np.zeros((13,7))
    avg_vel_hippocampus = np.zeros((13,7))

    avg_pressure             = np.zeros((13,7))
    avg_pressure_grey        = np.zeros((13,7))
    avg_pressure_white       = np.zeros((13,7))
    avg_pressure_amygdala    = np.zeros((13,7))
    avg_pressure_cortex      = np.zeros((13,7))
    avg_pressure_hippocampus = np.zeros((13,7))
    flux_out = np.zeros((13,7))
    t = np.zeros(13)
    # i compartmental index, k time index
    pmats = [avg_pressure, avg_pressure_grey, avg_pressure_white, avg_pressure_cortex, avg_pressure_hippocampus, avg_pressure_amygdala] 
    vel_mats = [avg_vel, avg_vel_grey, avg_vel_white, avg_vel_cortex, avg_vel_hippocampus, avg_vel_amygdala]
    transfer_mats = [transfer_matrix_parenchyma, transfer_matrix_grey, transfer_matrix_white, transfer_matrix_cortex, transfer_matrix_hippocampus, transfer_matrix_amygdala]



    folder = f"patients/{patient}"
    print("Starting plotting")
    for i, comp in enumerate(comps):
        XDMF_file = folder + f"/p_{comp}.xdmf"
        print(f"Reading {comp}")
        read = False
        p = []
        # Computing transfer between compartments
        for j, comp_ in enumerate(comps):
            print(comp_)
            if w[i,j] > 0:
                print(w[i,j])
                try:
                    print("Hei")
                    if recompute_transfer:
                        raise NameError()
                    for r, reg in enumerate(regions):
                        try:
                            print("filename")
                            filename = f"patients/{patient}/plots/transfer_{reg}_{compartment_names[i]}_{compartment_names[j]}.txt"

                            clean_data(filename,filename)
                            clean_data(filename,filename)
                            clean_data(filename, filename)
                            coeff = 1
                        except Exception:
                            filename = f"patients/{patient}/plots/transfer_{reg}_{compartment_names[j]}_{compartment_names[i]}.txt"
                            coeff = -1
                        with open(filename, 'r') as infile:
                            line = infile.readlines()[1].split()
                            line = line[1:]
                            transfer_mats[r][0,i,j] = coeff*float(line[0][1:])
                            for l in range(1, len(line)-1):
                                transfer_mats[r][l,i,j] = coeff*float(line[l])
                            transfer_mats[r][-1,i,j] = coeff*float(line[-1][:-1])
                except:
                    print(f"Failed {patient}, {comp}, {comp_}")
                    p2 = []
                    if not read:
                        for k in range(13):
                            p.append(read_xdmf(XDMF_file, mesh, 2, k, patient, comp))
                        print("Pressure read")
                        Volume_grey = assemble(1*dx(1,domain=mesh))
                        Volume_white = assemble(1*dx(2,domain=mesh))
                        Volume_amygdala = assemble(1*dxl(18,domain=mesh))
                        Volume_hippocampus = assemble(1*dxl(53,domain=mesh))
                        Volume_cortex = assemble(1*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))
                        print("Volume computed")

                    for k in range(13):
                        p2.append(read_xdmf(folder + f"/p_{comp_}.xdmf", mesh, 2, k, patient, comp))
                        read = True
                    print("Secondary pressure read")
                    for k in range(13):
                        transfer_matrix_parenchyma[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2[k] - p[k])*dx(mesh)) # (Pas)^-1 * Pa * mm^3 = mm^3/s = 1e-9 m^3/s = 1e-6 l/s 
                        transfer_matrix_grey[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2[k] - p[k])*dx(1,domain=mesh))
                        transfer_matrix_white[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2[k] - p[k])*dx(2,domain=mesh))
                        transfer_matrix_cortex[k,i,j] = 60/1000 * assemble(133.322*Constant(1)*w[i,j]*(p2[k] - p[k])*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))
                        transfer_matrix_hippocampus[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2[k] - p[k])*dxl(53,domain=mesh))
                        transfer_matrix_amygdala[k,i,j] =  60/1000 *assemble(133.322*Constant(1)*w[i,j]*(p2[k] - p[k])*dxl(18,domain=mesh))
                    print("Transfer computed")
        print("Checking vels")        
        try:                
            if recompute_velocities:
                raise NameError()
            for r, reg in enumerate(regions):
                filename = f"patients/{patient}/plots/avg_vel_{reg}_{compartment_names[i]}.txt"

                clean_data(filename,filename)
                clean_data(filename,filename)
                clean_data(filename, filename)
                with open(filename, 'r') as infile:
                    line = infile.readlines()[1].split()
                    line = line[1:]
                    vel_mats[r][0,i] = float(line[0][1:])
                    for l in range(1, len(line)-1):
                        vel_mats[r][l,i] = float(line[l])
                    vel_mats[r][-1,i] = float(line[-1][:-1])
        except:
            if not read:
                for k in range(13):
                    p.append(read_xdmf(XDMF_file, mesh, 2, k, patient, comp))
                Volume_grey = assemble(1*dx(1,domain=mesh))
                Volume_white = assemble(1*dx(2,domain=mesh))
                Volume_amygdala = assemble(1*dxl(18,domain=mesh))
                Volume_hippocampus = assemble(1*dxl(53,domain=mesh))
                Volume_cortex = assemble(1*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))                  
                read = True

            print(f"Failed {patient}, {comp}, vel")

            for k in range(13):
                avg_vel[k,i]             = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p[k]), grad(p[k]))**(0.5)*dx(mesh))/Volume #Units: grad(p) Pa/mm, kappa m^2/(sPa) dx: mm^3 - unit m^2/mm s ~ km/s
                avg_vel_grey[k,i]        = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p[k]), grad(p[k]))**(0.5)*dx(1,domain=mesh))/Volume_grey
                avg_vel_white[k,i]       = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p[k]), grad(p[k]))**(0.5)*dx(2,domain=mesh))/Volume_white
                avg_vel_amygdala[k,i]    = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p[k]), grad(p[k]))**(0.5)*dxl(18,domain=mesh))/Volume_amygdala
                avg_vel_cortex[k,i]      = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p[k]), grad(p[k]))**(0.5)*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))/Volume_cortex
                avg_vel_hippocampus[k,i] = 1e6*assemble(133.322*kappas[i]*Constant(1)*inner(grad(p[k]), grad(p[k]))**(0.5)*dxl(53,domain=mesh))/Volume_hippocampus
        print("Checking pressures")
        try:
            if recompute_pressure_avg:
                raise NameError()
            for r, reg in enumerate(regions):
                filename = f"patients/{patient}/plots/avg_pressure_{reg}_{compartment_names[i]}.txt"
                if recompute_pressure_avg:
                    a = float(filename)
                clean_data(filename,filename)
                clean_data(filename,filename)
                clean_data(filename, filename)
                with open(filename, 'r') as infile:
                    line = infile.readlines()[1].split()
                    line = line[1:]
                    pmats[r][0,i] = float(line[0][1:])
                    for l in range(1, len(line)-1):
                        pmats[r][l,i] = float(line[l])
                    pmats[r][-1,i] = float(line[-1][:-1])
        except Exception:
            if not read:
                for k in range(13):
                    p.append(read_xdmf(XDMF_file, mesh, 2, k, patient, comp))
                Volume_grey = assemble(1*dx(1,domain=mesh))
                Volume_white = assemble(1*dx(2,domain=mesh))
                Volume_amygdala = assemble(1*dxl(18,domain=mesh))
                Volume_hippocampus = assemble(1*dxl(53,domain=mesh))
                Volume_cortex = assemble(1*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))
            print(f"Failed {patient}, {comp}, pressure")
            for k in range(13):
                avg_pressure[k,i]             = assemble(p[k]*Constant(1)*dx(mesh))/Volume
                avg_pressure_grey[k,i]        = assemble(p[k]*Constant(1)*dx(1,domain=mesh))/Volume_grey
                avg_pressure_white[k,i]       = assemble(p[k]*Constant(1)*dx(2,domain=mesh))/Volume_white
                avg_pressure_amygdala[k,i]    = assemble(p[k]*Constant(1)*dxl(18,domain=mesh))/Volume_amygdala
                avg_pressure_cortex[k,i]      = assemble(p[k]*Constant(1)*(dxl(1002,domain=mesh) + dxl(1010,domain=mesh) + dxl(1023,domain=mesh)))/Volume_cortex
                avg_pressure_hippocampus[k,i] = assemble(p[k]*Constant(1)*dxl(53,domain=mesh))/Volume_hippocampus



            t[k] = k*200

        try:
            if int(mpi_comm.rank) == 0:
                os.system("mkdir " +folder + "/plots")
        except:
            pass
    
    regions = ["parenchyma", "grey matter", "white matter", "part of cortex", "hippocampus", "amygdala"]
    comps = ["a", "c", "v", "pa", "pc", "pv", "e"]
    compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]

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
                    outfile = f"patients/{patient}/plots/transfer_{regions[k]}_{compartment_names[i]}_{compartment_names[j]}"
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
                outfile =  f"patients/{patient}/plots/avg_vel_{regions[k%6]}_{compartment_names[i]}"
            else:
                plt.ylabel("Pressure [mmHg]")      
                plt.title(f"Average pressure in {regions[k%6]}")
                outfile =  f"patients/{patient}/plots/avg_pressure_{regions[k%6]}_{compartment_names[i]}"
            plt.legend(["Mean"])
            plt.savefig(outfile + ".pdf")
            plt.close()
            with open(outfile + ".txt", 'w+') as write_file:
                write_file.write(f"t {t}\n")
                write_file.write(f"other_avg {other_data[k][:,i]}\n")
            m += 1
    return transfer_matrix_parenchyma, transfer_matrix_grey, transfer_matrix_white, transfer_matrix_cortex, \
           transfer_matrix_hippocampus, transfer_matrix_amygdala, avg_pressure, avg_pressure_grey, \
           avg_pressure_white, avg_pressure_cortex, avg_pressure_amygdala, avg_pressure_hippocampus, \
           avg_pressure_amygdala, avg_vel, avg_vel_grey, avg_vel_white, avg_vel_cortex, avg_vel_hippocampus, \
           avg_vel_amygdala, ventricular_pressure_force, t, w



def compute_group_vals(group, recompute_transfer=True, recompute_velocities=True, recompute_pressure_avg=True):
    if group == "C":
        stop = 33
        start = 1
    else:
        stop = 14
        start = 1
    n = stop - start + 1
    transfer_matrix_parenchyma  = np.zeros((n,13, 7, 7))
    transfer_matrix_grey        = np.zeros((n,13, 7, 7))
    transfer_matrix_white       = np.zeros((n,13, 7, 7))
    transfer_matrix_cortex      = np.zeros((n,13, 7, 7))
    transfer_matrix_hippocampus = np.zeros((n,13, 7, 7))
    transfer_matrix_amygdala    = np.zeros((n,13, 7, 7))
    avg_pressure                = np.zeros((n,13,7))
    avg_pressure_grey           = np.zeros((n,13,7))
    avg_pressure_white          = np.zeros((n,13,7))
    avg_pressure_cortex         = np.zeros((n,13,7))
    avg_pressure_amygdala       = np.zeros((n,13,7))
    avg_pressure_hippocampus    = np.zeros((n,13,7))
    avg_pressure_amygdala       = np.zeros((n,13,7))
    avg_vel                     = np.zeros((n,13,7))
    avg_vel_grey                = np.zeros((n,13,7))
    avg_vel_white               = np.zeros((n,13,7))
    avg_vel_cortex              = np.zeros((n,13,7))
    avg_vel_hippocampus         = np.zeros((n,13,7))
    avg_vel_amygdala            = np.zeros((n,13,7))
    ventricular_pressure_force  = np.zeros((n,13,7))


    num_failed = 0
    failed_list = []

    for i in range(start, stop+1):
        try:
            transfer_matrix_parenchyma[i-1,:,:,:], transfer_matrix_grey[i-1,:,:,:], transfer_matrix_white[i-1,:,:,:], transfer_matrix_cortex[i-1,:,:,:], \
            transfer_matrix_hippocampus[i-1,:,:,:], transfer_matrix_amygdala[i-1,:,:,:], avg_pressure[i-1,:,:], avg_pressure_grey[i-1,:,:], \
            avg_pressure_white[i-1,:,:], avg_pressure_cortex[i-1,:,:], avg_pressure_amygdala[i-1,:,:], avg_pressure_hippocampus[i-1,:,:], \
            avg_pressure_amygdala[i-1,:,:], avg_vel[i-1,:,:], avg_vel_grey[i-1,:,:], avg_vel_white[i-1,:,:], avg_vel_cortex[i-1,:,:], avg_vel_hippocampus[i-1,:,:], \
            avg_vel_amygdala[i-1,:,:], ventricular_pressure_force[i-1,:,:], t, w = analyse_patient(f"{group}{i}",recompute_transfer, recompute_velocities, recompute_pressure_avg)
        except:
            num_failed += 1
            failed_list.append(i)
    transfer_data = [transfer_matrix_parenchyma, transfer_matrix_grey, transfer_matrix_white, transfer_matrix_cortex, \
        transfer_matrix_hippocampus, transfer_matrix_amygdala]
    other_data =  [avg_pressure, avg_pressure_grey, avg_pressure_white, avg_pressure_cortex, avg_pressure_hippocampus, \
        avg_pressure_amygdala, avg_vel, avg_vel_grey, avg_vel_white, avg_vel_cortex, avg_vel_hippocampus, avg_vel_amygdala]

    regions = ["parenchyma", "grey_matter", "white_matter", "cingulum", "hippocampus", "amygdala"]
    comps = ["a", "c", "v", "pa", "pc", "pv", "e"]
    compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]

    avg_transfer_data = []
    max_transfer_data = []
    min_transfer_data = []
    for i, data in enumerate(transfer_data):
        trans_mat = np.zeros((len(t),7,7))
        max_trans = np.zeros((len(t),7,7))
        min_trans = np.zeros((len(t),7,7))
        for j in range(7):
            for k in range(7):
                for l in range(len(t)):
                    avg = np.sum(data[:,l,j,k])/n
                    trans_mat[l,j,k] = avg
                max_trans_idx = np.argmax(np.sum(abs(data[:,:,j,k]), axis=1))
                min_trans_idx = np.argmin(np.sum(abs(data[:,:,j,k]), axis=1))
                max_trans[:,j,k] = data[max_trans_idx,: , j, k]
                min_trans[:,j,k] =data[min_trans_idx,: , j, k]
                
        avg_transfer_data.append(trans_mat)
        max_transfer_data.append(max_trans)
        min_transfer_data.append(min_trans)

    avg_other_data = []
    max_other_data = []
    min_other_data = []
    for i, data in enumerate(other_data):
        mat = np.zeros((len(t), 7))
        max_mat = np.zeros((len(t), 7))
        min_mat = np.zeros((len(t), 7))
        for j in range(7):
            for k in range(len(t)):
                avg = np.sum(data[:,k,j])/n
                mat[k,j] = avg
            max_idx = np.argmax(np.sum(abs(data[:,:,j]), axis=1))
            min_idx = np.argmin(np.sum(abs(data[:,:,j]), axis=1))
            max_mat[:,j] = data[max_idx,:,j]
            min_mat[:,j] = data[min_idx,:,j]
                
        avg_other_data.append(mat)
        max_other_data.append(max_mat)
        min_other_data.append(min_mat)

    m = 0
    try:
        os.system("mkdir results")
    except:
        pass
    try:
        os.system("mkdir results/plots/")
    except:
        pass


    folder = "results"

    for k, data in enumerate(avg_transfer_data):    
        for i in range(7):
            for j in range(i,7):
                if abs(w[i,j]) > 0:
                    plt.plot(t, data[:,i,j], t, max_transfer_data[k][:,i,j], t, min_transfer_data[k][:,i,j])
                    plt.xlabel("Time [s]")
                    plt.ylabel("Volume flux [(ml/min)/mm$^{-3}$]")
                    plt.title(f"Total transfer in {regions[k]}")
                    plt.legend(["Mean",f"Max {group}{max_idx+1}", f"Min {group}{min_idx+1}"])
                    outfile = folder + f"/plots/transfer_{regions[k]}_{compartment_names[i]}_{compartment_names[j]}_{group}"
                    plt.savefig(outfile + ".pdf")
                    plt.close()
                    with open(outfile + ".txt", 'w+') as write_file:
                        write_file.write(f"t {t}\n")
                        write_file.write(f"transfer_avg {data[:,i,j]}\n")
                        write_file.write(f"transfer_min {min_transfer_data[k][:,i,j]}\n")
                        write_file.write(f"transfer_max {max_transfer_data[k][:,i,j]}")
    for k, data in enumerate(avg_other_data):
        for i in range(7):            
            plt.plot(t, data[:,i],t, max_other_data[k][:,i], t, min_other_data[k][:,i])
            plt.xlabel("Time [s]")
            if m > 6:
                plt.ylabel("Speed [$\mu$m/s]")      
                plt.title(f"Average velocity in {regions[k%6]}")
                outfile = folder + f"/plots/avg_vel_{regions[k%6]}_{compartment_names[i]}_{group}"
            else:
                plt.ylabel("Pressure [mmHg]")      
                plt.title(f"Average pressure in {regions[k%6]}")
                outfile = folder + f"/plots/avg_pressure_{regions[k%6]}_{compartment_names[i]}_{group}"
            plt.legend(["Mean", f"Max {group}{max_idx+1}", f"Min {group}{min_idx+1}"])
            plt.savefig(outfile + ".pdf")
            plt.close()
            with open(outfile + ".txt", 'w+') as write_file:
                write_file.write(f"t {t}\n")
                write_file.write(f"other_avg {avg_other_data[k][:,i]}\n")
                write_file.write(f"other_min {min_other_data[k][:,i]}\n")
                write_file.write(f"other_max {max_other_data[k][:,i]}")
            m += 1

    with open("failed_pats.txt", 'w+') as outfile:
        outfile.write(f"Num fails : {num_failed}")
        outfile.write(f"Failed pats : {failed_list}")
    return avg_transfer_data, avg_other_data, t, w
        
def compare_groups(recompute_transfer=True, recompute_velocities=True, recompute_pressure_avg=True):
    avg_transfer_data_C, avg_other_data_C,t, w_C = compute_group_vals("C", recompute_transfer, recompute_velocities, recompute_pressure_avg)
    avg_transfer_data_NPH, avg_other_data_NPH, t, w_NPH = compute_group_vals("NPH", recompute_transfer, recompute_velocities, recompute_pressure_avg)    
    try:
        os.system("mkdir results/")
    except:
        pass
    try:
        os.system("mkdir results/C_vs_NPH/")
    except:
        pass
    folder = "results/C_vs_NPH"    
    m = 0
    regions = ["parenchyma", "grey matter", "white matter", "part of cortex", "hippocampus", "amygdala"]
    comps = ["a", "c", "v", "pa", "pc", "pv", "e"]
    compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]

    
    for i in range(7):    
        for k, data in enumerate(avg_transfer_data_C):
            for j in range(7):
                if abs(w_C[i,j]) > 0:
                    plt.plot(t, data[:,i,j])
                if abs(w_NPH[i,j]) > 0:
                    plt.plot(t, avg_transfer_data_NPH[k][:,i,j])
                if (abs(w_C[i,j]) > 0) or (abs(w_NPH[i,j]) > 0):
                    plt.xlabel("Time [s]")
                    plt.ylabel("Volume flux [ml/min]")
                    plt.title(f"Total transfer in {regions[k%6]}")
                    plt.legend(["Control mean", "iNPH mean"])
                    plt.savefig(folder + f"/transfer_{regions[k%6]}_{compartment_names[i]}_{compartment_names[j]}_C_NPH.pdf")
                    plt.close()
        for k, data in enumerate(avg_other_data_C):
            plt.plot(t, data[:,i], t, avg_other_data_NPH[k][:,i])
            plt.xlabel("Time [s]")
            if m > 6:
                plt.ylabel("Speed [$\mu$m/s]")      
                plt.title(f"Average velocity in {regions[k%6]}")
                outfile = folder + f"/avg_vel_{regions[k%6]}_{compartment_names[i]}_C_NPH.pdf"
            else:
                plt.ylabel("Pressure [mmHg]")      
                plt.title(f"Average pressure in {regions[k%6]}")
                outfile = folder + f"/avg_pressure_{regions[k%6]}_{compartment_names[i]}_C_NPH.pdf"
            plt.legend(["Control mean", "iNPH mean"])
            plt.savefig(outfile)
            plt.close()
            m += 1

if __name__ == '__main__':
    compare_groups(True, False, False)
