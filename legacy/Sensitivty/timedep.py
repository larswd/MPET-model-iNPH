import os
import matplotlib.pyplot as plt
import numpy as np
from analyse_data import read_data
from matplotlib.cm import get_cmap

os.system(f"mpirun -n 8 python3 MPET_brain.py -case C25 -res 32 -T 1800 -dt 15")
legends, p1 = read_data("C25", 32)

os.system(f"mpirun -n 8 python3 MPET_brain.py -case C25 -res 32 -T 1800 -dt 20")
legends, p2 = read_data("C25", 32)

os.system(f"mpirun -n 8 python3 MPET_brain.py -case C25 -res 32 -T 1800 -dt 45 ")
legends, p3 = read_data("C25", 32)

os.system(f"mpirun -n 4 python3 MPET_brain.py -case C25 -res 32 -T 1800 -dt 90")
legends, p4 = read_data("C25", 32)

os.system(f"mpirun -n 4 python3 MPET_brain.py -case C25 -res 32 -T 1800 -dt 120 ")
legends, p5 = read_data("C25", 32)

os.system(f"mpirun -n 4 python3 MPET_brain.py -case C25 -res 32 -T 1800 -dt 180 ")
legends, p6 = read_data("C25", 32)

 
titles = ["arterial", "capillary", "venous", "PVSa", "PVSc", "PVSv", "ECS"]
cmap = get_cmap('magma')
cols = cmap(np.linspace(0,1,81))

for i, title in enumerate(titles):
    legend = []
    plt.plot(np.linspace(0, 1800, len(p1[0,:])), p1[i+1,:], color=cols[1])
    plt.plot(np.linspace(0, 1800, len(p2[0,:])), p2[i+1,:], color=cols[11])
    plt.plot(np.linspace(0, 1800, len(p3[0,:])), p3[i+1,:], color=cols[21])
    plt.plot(np.linspace(0, 1800, len(p4[0,:])), p4[i+1,:], color=cols[31])
    plt.plot(np.linspace(0, 1800, len(p5[0,:])), p5[i+1,:], color=cols[51])
    plt.plot(np.linspace(0, 1800, len(p6[0,:])), p6[i+1,:], color=cols[61])      
    legend.append("$\Delta t=15$ s")
    legend.append("$\Delta t=20$ s")
    legend.append("$\Delta t=45$ s")
    legend.append("$\Delta t=90$ s")
    legend.append("$\Delta t=120$ s")
    legend.append("$\Delta t=180$ s")
                          
    plt.grid()
    plt.xlim([0,1800])
    plt.ylim([0,1.1*np.amax(p1[i+1,:])])
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [mmHg]")
    plt.title(f"Pressure in {title} compartment")
    plt.legend(legend)
    plt.savefig(f"diff_{title}_C25.pdf")
    plt.show()