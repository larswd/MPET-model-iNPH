from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from IPython import embed
import sys

set_log_level(30)

#os.system("dijitso clean")
# Leser kommmandolinjeargumenter
parser = argparse.ArgumentParser()
parser.add_argument("-T","--Time", type=float, default=3000, 
                    help="Length of diffusion experiment in seconds")
parser.add_argument("-dt", "--time_step", type=float, default=20)
parser.add_argument("-Q_in", "--blood_flow", type=float, default=712.5)
parser.add_argument("-a", "--alpha", type=float, default=1)
args = parser.parse_args()

T = args.Time
dt = args.time_step

alpha = args.alpha
patient = "C25"
res = 32
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

def compute_Rout(mesh, alpha,T=T,dt=dt):

    geo = mesh.ufl_cell()


    P1 = FiniteElement('CG',geo,2)
    R = FunctionSpace(mesh, P1)

    # PHYSICAL PARAMETERS (hentet fra Vegards skript)
    r_factor = 133*1e6*60*alpha     # mmHg/(mL/min) = 133 Pa/(1e-3 L / 60 s) = 133 Pa / (1e-3 * 1e-3 m^3/ 60s) = 133*1e6*60 Pa/s



    Qprod = Expression("5.787e-5", degree=0)

    t = 0


    def Compliance(p,E = 0.2e6, p0 = 9*133.33):
        return 1./(E*(p-p0))

    CSF_prod = 0.33*1e3/60.
    CSF_infusion = 1.5*1e3/60.
    CSF_in = CSF_prod+CSF_infusion
    print(1e-9*CSF_in)

    surface_area = assemble(interpolate(Constant(1), R)*(ds(1) + ds(2) + ds(3))) 

    blood_flow = args.blood_flow
    b_avg = blood_flow*1e3/60./surface_area # mL/min = cm^3/min = 1e3*mm^3/(60s)
    b_in = Constant(b_avg)


    p_AG = 8.4*133.33           # AG pressure
    p_CSF = Constant(9.2*133.33)  # CSF pressure
    p_crib = Constant(0)        # cribriform plate pressure


    t = 0


    R_AG = 10.81*r_factor
    R_crib = 67*r_factor

    p_vals = []
    t_vals = []
    while t < T:

        print("t = ", t)
        # ODE for nytt trykk. Eksplisitt diskretisering.
        if t < 700:
            p_CSF_new = float(p_CSF) + dt/Compliance(float(p_CSF))*(CSF_prod*1e-9 - 1/R_AG*(float(p_CSF) - p_AG) - 1/R_crib*(float(p_CSF) - p_crib))
            p_0 = float(p_CSF_new)/133.322
        else:
            p_CSF_new = float(p_CSF) + dt/Compliance(float(p_CSF))*(CSF_in*1e-9 - 1/R_AG*(float(p_CSF) - p_AG) - 1/R_crib*(float(p_CSF) - p_crib))
            p_pl = float(p_CSF_new)/133.322
        p_CSF.assign(p_CSF_new)
        p_vals.append(float(p_CSF))
        t_vals.append(t)
        print("\nP_CSF = ", float(p_CSF)/133.322, ' mmHg')
        t += dt
    R0 = (p_pl-p_0)/(1.5)
    plt.plot(t_vals, p_vals)
    plt.show()
    print(R0)
    return R0

alphs = [1]#np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])

R0s = np.zeros(len(alphs))

for i, alpha in enumerate(alphs):
    R0s[i] = compute_Rout(mesh, alpha)

Ravg = np.sum(R0s)/len(R0s)
alphavg = np.sum(alphs)/len(alphs)

m = np.sum((R0s-Ravg)*(alphs-alphavg))/np.sum((alphs-alphavg)**2)
b = np.sum(R0s - m*alphs)/len(alphs)
print(b, m)

a = np.linspace(0, 3, 70)
f = lambda x: m*x + b

MEDIUM_SIZE = 12
SMALL_SIZE = 10
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)

plt.plot(a, f(a), 'r')
plt.plot(alphs, R0s, 'bo', alpha=0.5)
plt.grid()
plt.xlim([0,3])
plt.ylim([0,30])
plt.xlabel("$\\alpha$")
plt.ylabel("$R_{out}$")
const_term = ""
if b > 0:
    const_term += " + %.2f" % abs(b)
elif b == 0:
    const_term = ""
else:
    const_term += " - %.2f" % abs(b)
leg ="$f(\\alpha) = %.2f\\alpha$" % m 
leg += const_term
plt.legend([leg, "Computed"])
plt.title("Resistance vs $\\alpha$-parameter")
plt.show()
