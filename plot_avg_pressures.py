import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.cm import get_cmap
import matplotlib

def read_data(patient, resolution, file="pressure_average.dat"):
    with open(f"patients/{patient}/results{resolution}/{file}", 'r') as infile:
        lines = infile.readlines()
        legends = lines[0].split()
        p_matrix = np.zeros((len(legends),len(lines[1:])))
        for i in range(len(p_matrix[0,:])):
            line = lines[i+1].split()
            for j,x in enumerate(line):
                p_matrix[j,i] = float(x)
    return legends, p_matrix

ignore_list = [8,24]

start_c = int(sys.argv[1])
stop_c  = int(sys.argv[2])

start_n = int(sys.argv[3])
stop_n  = int(sys.argv[4])

try:
    filename = sys.argv[5]
except:
    filename = "pressure_average.dat"
    reg = "parenchyma"
    idx = 0

if filename == "1.txt":
    idx = 1
    reg = "grey_matter"
elif filename == "2.txt":
    idx = 2
    reg = "white_matter"
else:
    pass

first = True
k = 0

#Control
sample_size_c = stop_c-start_c+1
for i in range(start_c, stop_c+1):
    if first:
        print(f"C{i}")
        first = False
        legends, pmat1 = read_data(f"C{i}", 32, filename)
        pmat_c = np.zeros((sample_size_c, len(pmat1[:,0]), len(pmat1[0,:])))
        pmat_c[k,:,:] = pmat1
        k += 1
    elif i in ignore_list:
        pass
    else:
        print(f"C{i}")
        legends, pmat_c[k,:,:] = read_data(f"C{i}", 32, filename)
        k += 1 
pmat_c = pmat_c[:-2,:,:]
sample_size_c -= 2
t_avg_pressures_c = np.zeros((sample_size_c, 7))
pat_avg_pressures_c = np.zeros((7, len(pmat_c[0,0,:])))


for i in range(sample_size_c ):
    for j in range(7):
        t_avg_pressures_c[i,j] = np.sum(pmat_c[i,j+1,:])/1800


#NPH
first = True
k = 0   
sample_size_n = stop_n-start_n+1
for i in range(start_n, stop_n+1):
    if first:
        print(f"NPH{i}")
        first = False
        legends, pmat1 = read_data(f"NPH{i}", 32, filename)
        pmat_n = np.zeros((sample_size_n, len(pmat1[:,0]), len(pmat1[0,:])))
        pmat_n[k,:,:] = pmat1
        k += 1
    else:
        print(f"NPH{i}")
        legends, pmat_n[k,:,:] = read_data(f"NPH{i}", 32, filename)
        k += 1 

t_avg_pressures_n = np.zeros((sample_size_n, 7))
pat_avg_pressures_n = np.zeros((7, len(pmat_n[0,0,:])))


for i in range(sample_size_n):
    for j in range(7):
        t_avg_pressures_n[i,j] = np.sum(pmat_n[i,j+1,:])/1800


region_names = ["parenchyma", "grey matter", "white matter"]
comp_names = ["Arterial", "Capillary", "Venous", "Arterial PVS", "Capillary PVS", "Venous PVS", "ECS"]

cmap = get_cmap('magma')
cols = cmap(np.linspace(0,1,15))
k = 4

SMALL_SIZE = 15
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE+1)
for i in range(7):
    print(i)    
    fig, axs = plt.subplots(1,2,gridspec_kw = {'wspace':0.05})
    fig.suptitle(f"{comp_names[i]} pressure in the {region_names[idx]}")

    pat_avg_pressures_c[i,:] = np.sum(pmat_c[:,i+1, :], axis=0)/sample_size_c
    pat_avg_pressures_n[i,:] = np.sum(pmat_n[:,i+1, :], axis=0)/sample_size_n
    axs[0].plot(pmat_c[0,0,:], pat_avg_pressures_c[i,:], color=cols[k])
    axs[1].plot(pmat_c[0,0,:], pat_avg_pressures_n[i,:], color=cols[k])
    

    lowest_p_c =  np.zeros(len(pmat_c[0, 0, :]))
    highest_p_c = np.zeros(len(pmat_c[0, 0, :]))
    
    lowest_p_n =  np.zeros(len(pmat_c[0, 0, :]))
    highest_p_n = np.zeros(len(pmat_c[0, 0, :]))
    
    for j in range(len(pmat_c[0,0,:])):
        lowest_p_c[j]  = np.amin(pmat_c[:,i+1,j])
        lowest_p_n[j]  = np.amin(pmat_n[:,i+1,j])
        highest_p_c[j] = np.amax(pmat_c[:,i+1,j])
        highest_p_n[j] = np.amax(pmat_n[:,i+1,j])

    t = np.linspace(0, 2500, len(lowest_p_c))
    axs[0].plot(t, lowest_p_c, linestyle='dotted', color=cols[k], alpha=0.6)
    axs[0].plot(t, highest_p_c,linestyle='dotted', color=cols[k], alpha=0.6)
    axs[0].fill_between(t, lowest_p_c, highest_p_c, color=cols[k], alpha=0.5)
    axs[1].plot(t, lowest_p_n, linestyle='dotted', color=cols[k], alpha =0.6)
    axs[1].plot(t, highest_p_n,linestyle='dotted', color=cols[k], alpha = 0.6)
    axs[1].fill_between(t, lowest_p_n, highest_p_n, color=cols[k], alpha=0.5)
    
    axs[0].set_title("Control")
    axs[1].set_title("NPH")

    # Shrinking and moving plot to better position
    for j in range(2):
        box = axs[j].get_position()
        axs[j].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        axs[j].set_xlabel("Time [s]")
        axs[j].grid(alpha=0.7)        

    maximum = max(highest_p_c[-1], highest_p_n[-1])
    for ax in axs:
        ax.set_xlim([0,2500])
        ax.set_ylim(bottom=0, top = 1.1*abs(maximum))
        ax.plot(600*np.ones(200), np.linspace(0,1.1*(abs(maximum)),200), linestyle='dashed', color="red", alpha=0.7, linewidth=0.8)
    axs[1].axes.yaxis.set_ticklabels([])
    axs[0].set_ylabel("Pressure [mmHg]")
    print("\n")
    print("reg no: %d" %i)
    print("before")
    print(pat_avg_pressures_c[i,0], pat_avg_pressures_n[i,0])

    print("after")
    print(pat_avg_pressures_c[i,-1], pat_avg_pressures_n[i,-1])
    print("Highest")
    print(highest_p_c[-1], highest_p_n[-1])
    print("Lowest")
    print(lowest_p_c[-1], lowest_p_n[-1])
    plt.savefig(f"pressure_compare_{legends[i+1]}_{reg}.pdf")
    plt.close()
