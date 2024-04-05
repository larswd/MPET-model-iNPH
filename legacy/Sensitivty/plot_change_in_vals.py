from analyse_data import *
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.cm import get_cmap
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("-T","--Time", type=float, default=1800, 
                    help="Length of diffusion experiment in seconds")
parser.add_argument("-dt", "--time_step", type=float, default=20)
parser.add_argument("-Q_in", "--blood_flow", type=float, default=712.5)
args = parser.parse_args()

res_list = [8, 16,32,64, 96]
orders   = [1,2]
first = True

for i,res in enumerate(res_list):
    for j, order in enumerate(orders):
        if first:
            os.system(f"mpirun -n 16 python3 MPET_brain.py -T {args.Time} -dt {args.time_step} -res {res} -case partial_C25 --order {order}")
            legends, pmat = read_data("partial_C25", res)
            data_mat = np.zeros((len(res_list), len(orders), len(pmat[:,0]), len(pmat[0,:])))
            data_mat[i,j,:,:] = pmat
            first = not first
        if (res <40 or order < 1.5) and not first:
            os.system(f"mpirun -n 16 python3 MPET_brain.py -T {args.Time} -dt {args.time_step} -res {res} -case partial_C25 --order {order}")
            legends, data_mat[i,j,:,:] = read_data("partial_C25", res)

cmap = get_cmap('magma')
cols = cmap(np.linspace(0,1,71))

comp_names = ["arteries", "capillaries", "veins", "arterial PVS", "capillary PVS", "venous PVS", "ECS"]

for j in range(1, 8):
    legend = []
    for i in range(len(res_list)-2):
        plt.plot(data_mat[i,0,0,:]-20, data_mat[i,0,j,:], linestyle="--", color=cols[10*i])
        plt.plot(data_mat[i,1,0,:]-20, data_mat[i,1,j,:], color=cols[10*i])

        legend.append(f"CG1 RP{res_list[i]}")
        legend.append(f"CG2 RP{res_list[i]}")
        k = i
    plt.plot(data_mat[i,0,0,:]-20, data_mat[-2,0,j,:], linestyle="--", color = cols[10*(k+1)])
    legend.append(f"CG1 RP64")
    plt.plot(data_mat[i,0,0,:]-20, data_mat[-1,0,j,:], linestyle="--", color=cols[10*(k+2)])
    legend.append("CG1 RP96")
    plt.grid()
    plt.xlabel("Time [s]")
    plt.xlim([0,1800])
    plt.ylabel("Presure [mmHg]")
    plt.ylim([0,1.1*np.amax(data_mat[:,:,j,:])])
    plt.legend(legend)
    plt.title("Average pressure in the " + comp_names[j-1])
    plt.savefig(f"../pressure_{legends[j]}_CG1_CG2.pdf")
    plt.close()