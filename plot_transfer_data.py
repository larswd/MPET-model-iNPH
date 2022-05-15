from matplotlib.cm import get_cmap
from IPython import embed
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from clean_data import clean_transfer_data
import os
from os.path import exists


connected = np.array([[0,1,0,1,0,0,0],
                       [1,0,1,0,1,0,0],
                       [0,1,0,0,0,1,0],
                       [1,0,0,0,1,0,1],
                       [0,1,0,1,0,1,1],
                       [0,0,1,0,1,0,1],
                       [0,0,0,1,1,1,0]])

regions = ["parenchyma", "grey_matter", "white_matter", "cingulum", "amygdala", "hippocampus"]
region_names = ["parenchyma", "grey matter", "white matter", "cingulum", "amygdala", "hippocampus"]
comps = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
comp_names = ["Arterial", "Capillary", "Venous", "Arterial PVS", "Capillary PVS", "Venous PVS", "ECS"]
groups = ["C", "NPH"]



def clean_patient_transfer_data(patient,region, comp, comp2, group, loc="patients/"):
    #os.system(f"mmv patients/{patient}/plots/'transfer_part of cortex_*' patients/{patient}/plots/transfer_cingulum_#1")
    #os.system(f"mmv patients/{patient}/plots/'transfer_grey matter_*' patients/{patient}/plots/transfer_grey_matter_#1")
    #os.system(f"mmv patients/{patient}/plots/'transfer_white matter_*' patients/{patient}/plots/transfer_white_matter_#1")
    
    with open(loc + f"{patient}/transfer_{region}_{comp}_{comp2}.txt", 'r') as infile:
        lines = infile.readlines()
        filename = f"transfer_{region}_{comp}_{comp2}.txt"
        j = 0
        for i,line in enumerate(lines):
            completed = False
            k = i
            while not completed:   
                if k + 1 < len(lines):
                    if lines[i+1][0] == 't':
                        completed = True
                        if lines[i][-2:] == "]\n":
                            if lines[i][-3] == " ":
                                lines[i] = lines[i][:-3] + lines[i][-2:]
                                completed = False
                            pass
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
    
    if lines[-1][0] != "t":
        lines[-2] = lines[-2][:-1] + lines[-1] + "\n"
        lines = lines[:-1]
    with open(loc + f"{patient}/transfer_{region}_{comp}_{comp2}.txt", 'w') as outfile:
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

def recompute_avgs(ignore_list, group, clean=False):
    if group == "C":
        stop = 33
    else:
        stop = 14
    first = True
    start = 1
    regions = ["parenchyma", "grey_matter", "white_matter", "amygdala", "cingulum", "hippocampus"]
    compartments = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]

    for i in range(start, stop+1):
        for l, region in enumerate(regions):
            for j,comp in enumerate(comps):
                connections = connected[j]
                for k,comp2 in enumerate(comps[j:]):
                    k += j
                    if connections[k]:

                        try:
                            if exists(f"txtfiles/{group}{i}/transfer_{region}_{comp}_{comp2}.txt"):
                                filename = f"txtfiles/{group}{i}/transfer_{region}_{comp}_{comp2}.txt"
                                a = 1
                                if clean:
                                    clean_patient_transfer_data(f"{group}{i}",region, comp, comp2, group, loc="txtfiles/")
                                    clean_patient_transfer_data(f"{group}{i}",region, comp, comp2, group, loc="txtfiles/")
                                    clean_patient_transfer_data(f"{group}{i}",region, comp, comp2, group, loc="txtfiles/")
                            else:
                                filename = f"txtfiles/{group}{i}/transfer_{region}_{comp2}_{comp}.txt"
                                a = -1
                                if clean:
                                    clean_patient_transfer_data(f"{group}{i}",region, comp, comp2, group, loc="txtfiles/")
                                    clean_patient_transfer_data(f"{group}{i}",region, comp, comp2, group, loc="txtfiles/")
                                    clean_patient_transfer_data(f"{group}{i}",region, comp, comp2, group, loc="txtfiles/")
                            
                            with open(filename, 'r') as infile:

                                lines = infile.readlines()
                                line = lines[0].split()
                                if first:
                                    time = np.zeros(len(line)-1)
                                    for m,tval in enumerate(line[2:-1]):
                                        time[m+1] = float(tval)
                                    time[0] = float(line[2])
                                    time[-1] = float(line[-1][:-1])
                                    tmat = np.zeros((stop-start+1, len(regions),7,7,len(time)))
                                    first = False
                                line = lines[1].split()
                                transfers = np.zeros(len(time))
                                transfers[0] = a*float(line[1][1:])
                                for m,tval in enumerate(line[2:-1]):
                                    transfers[m+1] = a*float(tval)
                                transfers[-1] = a*float(line[-1][:-1])
                                if abs(transfers[0]) < 1e-20 and region == regions[0] and j < 3 and connected[j,k] > 0 and k < 3:
                                    print("\nZero transfer at t=0")
                                    print(filename)
                                if abs(transfers[-1]) < 1e-20 and region == regions[0] and j < 3 and connected[j,k] > 0 and k < 3:
                                    print("\nZero transfer at t=2400")
                                    print(filename)                                    
                                tmat[i-1,l,j,k,:] = transfers
                        except:
                            pass
    
    c = 0
    for i in ignore_list:
        i -= c
        c += 1
        temp = tmat[:(i-1),:,:,:,:]
        temp2 = tmat[i:,:,:,:,:]
        tmat = np.zeros((len(tmat[:,0,0,0,0])-1,len(regions),7,7,len(time)))
        for j in range(len(temp[:,0,0,0,0])):
            tmat[j,:,:,:,:] = temp[j,:,:,:,:]
            n = j
        for j in range(len(temp2[:,0,0,0,0])):
            tmat[n+j+1,:,:,:,:] = temp2[j,:,:,:,:]
        

    compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
    avg_transfer = np.zeros((len(regions),7,7,len(time)))
    for i,region in enumerate(regions):
        for j, comp in enumerate(comps):
            connections = connected[j]
            for k, comp2 in enumerate(comps[j:]):
                k += j
                if connections[k]:
                    for t in range(len(time)):
                        avg_transfer[i,j,k,t] = np.sum(tmat[:,i,j,k,t])/(len(tmat[:,i,j,k,t]))
                outfile_name =  f"txtfiles/avg_transfer_{regions[i]}_{compartment_names[j]}_{compartment_names[k]}_{group}.txt"
                if connections[k]:
                    if region == regions[0] and j < 3 and k < 3:
                        print(tmat[:,i,j,k,0])
                        print(tmat[:,i,j,k,-1])
                    min_transfer = np.zeros(tmat[0,i,j,k,:].shape)
                    max_transfer = np.zeros(tmat[0,i,j,k,:].shape)
                    for t in range(len(min_transfer)):
                        min_transfer[t] = np.amin(tmat[:,i,j,k,t])
                        max_transfer[t] = np.amax(tmat[:,i,j,k,t])
                    #print(f"\nmax_transfer_idx = {max_transfer_idx}, region {region}, comp1 {comp}, comp2 {comp2}, group = {group}")

                    #print(f"\nmin_transfer_idx = {min_transfer_idx}, region {region}, comp1 {comp}, comp2 {comp2}, group = {group}")

                    with open(outfile_name, 'w+') as outfile:
                        outfile.write(f"t {time}\n")
                        outfile.write(f"transfer_avg {avg_transfer[i,j,k,:]}\n")
                        outfile.write(f"transfer_min {min_transfer}\n")
                        outfile.write(f"transfer_max {max_transfer}\n")





def read_transfer_data(region, comp, comp2, group):
    n = 0
    for i in range(3):
        if os.path.isfile(f'txtfiles/avg_transfer_parenchyma_{comp}_{comp2}_{group}.txt'):
            clean_transfer_data(region, comp, comp2, group)
        else:
            clean_transfer_data(region, comp2, comp, group)
    with open(f"txtfiles/avg_transfer_{region}_{comp}_{comp2}_{group}.txt", 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            line = line.split()
            if int(n) == 0:
                t = np.zeros(13)
                for i in range(13):
                    t[i] = i*200
                y = np.zeros(13)
                ymin = y.copy()
                ymax = y.copy()
                n +=1
            elif int(n) == 1:
                y[0] = float(line[1][1:])
                for i in range(1,len(t)-1):
                    y[i] = float(line[1+i])
                y[-1] = float(line[-1][:-1])
                n += 1
            elif int(n) == 2:
                ymin[0] = float(line[1][1:])
                for i in range(1,len(t)-1):
                    ymin[i] = float(line[1+i])
                ymin[-1] = float(line[-1][:-1])
                n += 1
            else:
                ymax[0] = float(line[1][1:])
                for i in range(1,len(t)-1):
                    ymax[i] = float(line[1+i])
                ymax[-1] = float(line[-1][:-1])
                #embed()
    return t, y, ymin, ymax


if __name__ == '__main__':
    reg = int(sys.argv[1]) - 1
    com = int(sys.argv[2]) - 1
    group_names  = ["Control", "NPH"]

    #recompute_avgs([8,24], "C", True)
    #
    #recompute_avgs([], "NPH", True)
    connected = connected[com,:]
    n_e = np.sum(connected)
    first = True

    # PVSA -> ECS -> PVSV
    cmap = get_cmap('cividis')
    cols = cmap(np.linspace(0,1,2*n_e + 2))
    SMALL_SIZE = 15
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE+1)

    fig, axs = plt.subplots(1,2,gridspec_kw = {'wspace':0.05})
    fig.suptitle(f"Total fluid exchange with {comp_names[com]} in {region_names[reg]}")
    maxval = []
    minval = []
    for j,group in enumerate(groups):
        k = 1
        legend = []
        for i, edge in enumerate(connected):
            if edge:
                try:
                    t,y, ymin, ymax = read_transfer_data(regions[reg], comps[com], comps[i], group)
                except Exception:
                    t,y, ymin, ymax = read_transfer_data(regions[reg], comps[i], comps[com], group)
                    y = -y
                    ymin = -ymin
                    ymax = -ymax
                t = np.linspace(0,2500, len(y))
                axs[j].plot(t,y, color=cols[k])
                axs[j].plot(t, ymin, label='_nolegend_', linestyle='dotted',color=cols[k])
                axs[j].plot(t, ymax, label='_nolegend_', linestyle='dotted',color=cols[k])
                axs[j].fill_between(t,ymin,ymax,color=cols[k], alpha=0.2)
                axs[j].set_title(group_names[j])
                legend.append(comp_names[i])
                k+=2
                maximum = max(abs(ymax[-1]), abs(ymin[-1]))
                minimum = min(ymax[-1], ymin[-1])
                
                if len(maxval) < j + 1:
                    maxval.append(maximum)
                    minval.append(minimum)
                elif maximum > maxval[j]:
                    maxval[j] = maximum
                if minimum < minval[j]:
                    minval[j] = minimum
                print(f"\nBefore {group} - {comps[i]}")
                print(y[2], ymax[2], ymin[2])
                print(f"\nAfter {group} - {comps[i]}")
                print(y[-1], ymax[-1], ymin[-1] )

        
        box = axs[j].get_position()
        axs[j].set_position([box.x0, box.y0 + box.height * 0.1,
                box.width, box.height * 0.9])
        axs[j].set_xlabel("Time [s]")
        axs[j].grid(alpha=0.7)
        
    low = min(1.2*min(minval[0], minval[1]), 0.8*min(minval[0], minval[1]))
    top = 1.1*abs(max(maxval[0], maxval[1]))
    for ax in axs:
        ax.set_ylim(bottom=low, top = top)
        ax.plot(600*np.ones(200), np.linspace(low,top,200), linestyle='dashed', color="red", alpha=0.7, linewidth=0.8)
        ax.set_xlim([0,2500])
    axs[1].axes.yaxis.set_ticklabels([])
    axs[1].legend(legend, loc='upper center', bbox_to_anchor=(-0.03, -0.14),fancybox=True, shadow=True, ncol=sum(connected))
    axs[0].set_ylabel("Volume flux [ml/min]")
    plt.savefig(f"figs/tot_transfer_{comps[com]}_{regions[reg]}.pdf", bbox_inches='tight')
    plt.show()