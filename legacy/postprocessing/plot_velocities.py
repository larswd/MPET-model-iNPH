from matplotlib.cm import get_cmap
from IPython import embed
import sys
import matplotlib.pyplot as plt
import numpy as np
from clean_data import clean_velocity_data
import os
import matplotlib

regions = ["parenchyma", "grey_matter", "white_matter", "cingulum", "amygdala", "hippocampus"]
region_names = ["parenchyma", "grey matter", "white matter", "cingulum", "amygdala", "hippocampus"]
comps = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
groups = ["C", "NPH"]
comp_names = ["Arterial", "Capillary", "Venous", "Arterial PVS", "Capillary PVS", "Venous PVS", "ECS"]
CBVf = 0.033 #Barbacaru
porosity = [0.33*CBVf, 0.1*CBVf, 0.57*CBVf]
porosity.append(1.4*porosity[0])
porosity.append(1*porosity[1])
porosity.append(1.4*porosity[2])
porosity.append(0.14)

def clean_patient_velocity_data(patient,region, comp, group, loc="patients/"):
    #os.system(f"mmv patients/{patient}/plots/'avg_vel_part of cortex_*' patients/{patient}/plots/transfer_cingulum_#1")
    #os.system(f"mmv patients/{patient}/plots/'avg_vel_grey matter_*' patients/{patient}/plots/transfer_grey_matter_#1")
    #os.system(f"mmv patients/{patient}/plots/'avg_vel_white matter_*' patients/{patient}/plots/transfer_white_matter_#1")
    print(patient, region, comp)
    
    
    with open(loc + f"{patient}/avg_vel_{region}_{comp}.txt", 'r') as infile:
        lines = infile.readlines()

        filename = f"avg_vel_{region}_{comp}.txt"
        j = 0
        for i,line in enumerate(lines):
            completed = False
            k = i
            while not completed:   
                if k + 1 < len(lines):
                    if lines[i+1][0] == 't' or lines[i+1][0] == 'o':
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
            print(1)
            if lines[-1][-3] == " ":
                lines[-1] = lines[-1][:-3] + lines[-1][-2:]
                completed = True
            pass
        elif lines[-1][-1] == "\n" and (lines[-1][-2:] != "]\n" and lines[-1][-2:] != " \n"):
            print(2)
            lines[-1] = lines[-1][:-1]
            lines[-1] += "]\n"
        elif lines[-1][-1] == "\n" and lines[-1][-2] == " ":
            print(3)
            lines[-1] = lines[-1][:-2] + "\n"
            last_line_white_space = True
        elif lines[-1][-2:] == " ]":
            lines[-1] = lines[-1][:-2] + lines[-1][-1]
            last_line_white_space = True
        else:
            lines[-1] += "]\n"
    
    if lines[-1][0] != "t" and lines[-1][0] != "o":
        lines[-2] = lines[-2][:-1] + lines[-1] + "\n"
        lines = lines[:-1]
    with open(loc + f"{patient}/avg_vel_{region}_{comp}.txt", 'w') as outfile:
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
                if clean and not i in ignore_list:
                    clean_patient_velocity_data(f"{group}{i}",region, comp, group, loc="txtfiles/")
                    clean_patient_velocity_data(f"{group}{i}",region, comp, group, loc="txtfiles/")
                    clean_patient_velocity_data(f"{group}{i}",region, comp, group, loc="txtfiles/")
                try:
                    with open(f"txtfiles/{group}{i}/avg_vel_{region}_{comp}.txt", 'r') as infile:
                        lines = infile.readlines()
                        line = lines[0].split()
                        if first:
                            time = np.zeros(len(line)-1)
                            for m,tval in enumerate(line[2:-1]):
                                time[m+1] = float(tval)
                            time[0] = float(line[1][1:])
                            time[-1] = float(line[-1][:-1])
                            vmat = np.zeros((stop-start+1, len(regions),7,len(time)))
                            first = False
                        line = lines[1].split()
                        vels = np.zeros(len(time))
                        vels[0] = float(line[1][1:])
                        for m,tval in enumerate(line[2:-1]):
                            vels[m+1] = float(tval)
                        vels[-1] = float(line[-1][:-1])
                        vmat[i-1,l,j,:] = vels
                except:
                    print(f"{group}{i}")
                    pass
    
    c = 0
    for i in ignore_list:
        i -= c
        c += 1
        temp = vmat[:(i-1),:,:,:]
        temp2 = vmat[i:,:,:,:]
        vmat = np.zeros((len(vmat[:,0,0,0])-1,len(regions),7,len(time)))
        for j in range(len(temp[:,0,0,0])):
            vmat[j,:,:,:] = temp[j,:,:,:]
            n = j
        print(n)
        for j in range(len(temp2[:,0,0,0])):
            vmat[n+j+1,:,:,:] = temp2[j,:,:,:]
    
    min_vel = np.zeros((len(regions), 7, len(time)))
    max_vel = np.zeros((len(regions), 7, len(time)))  
    compartment_names = ["arterial", "capillary", "venous", "PVS_a", "PVS_c", "PVS_v", "ISF"]
    avg_vel = np.zeros((len(regions),7,len(time)))
    for i,region in enumerate(regions):
        for j, comp in enumerate(comps):
            for t in range(len(time)):
                avg_vel[i,j,t] = np.sum(vmat[:,i,j,t])/(len(vmat[:,i,j,t]))
                outfile_name =  f"txtfiles/avg_vel_{regions[i]}_{compartment_names[j]}_{group}.txt"
                

                min_vel[i,j,t]     = np.amin(vmat[:,i,j,t])
                max_vel[i,j,t]     = np.amax(vmat[:,i,j,t])
                
            with open(outfile_name, 'w+') as outfile:
                outfile.write(f"t {time}\n")
                outfile.write(f"other_avg {avg_vel[i,j,:]}\n")
                outfile.write(f"other_min {min_vel[i,j,:]}\n")
                outfile.write(f"other_max {max_vel[i,j,:]}\n")





def read_velocity_data(region, comp, group):
    n = 0
    for i in range(3):
        clean_velocity_data(region, comp, group)
    print(f"txtfiles/avg_vel_{region}_{comp}_{group}.txt")
    with open(f"txtfiles/avg_vel_{region}_{comp}_{group}.txt", 'r') as infile:
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

    return t, y, ymin, ymax


if __name__ == '__main__':
    com = int(sys.argv[1]) - 1
    first = True
    group_names = ["Control", "NPH"]

    #recompute_avgs([8,24], "C", True)
    #recompute_avgs([], "NPH", True)
    # PVSA -> ECS -> PVSV
    cmap = get_cmap('inferno')
    cols = cmap(np.linspace(0,1,len(regions)))

    regions = ["parenchyma", "grey_matter", "white_matter"]
    region_names = ["parenchyma", "grey matter", "white matter"]
    SMALL_SIZE = 15
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE+1)
    
    for i, region in enumerate(regions):
        
        # Making subplot
        fig, axs = plt.subplots(1,2,gridspec_kw = {'wspace':0.05})

        # Index to get right colour
        k = 3
        # Main title
        fig.suptitle(f"{comp_names[com]} fluid velocity in {region_names[i]}")
        
        # List for finding highest and lowest value
        maxval = []
        for j,group in enumerate(groups):
            t,y, ymin, ymax = read_velocity_data(region, comps[com], group)
            t = np.linspace(0,2500, len(y))
            # Scaling result
            y = y/porosity[com]
            ymin = ymin/porosity[com]
            ymax = ymax/porosity[com]
            print(group)
            print("\nStart")
            
            print(y[0])
            print(ymax[0])
            print(ymin[0])

            print("\nEnd")
            print(y[-1])
            print(ymax[-1])
            print(ymin[-1])
            print("\n")
            # Plotting main result with highest and lowest value
            if com < 3:
                axs[j].plot(t, y/1e6, color=cols[k])
                axs[j].plot(t, ymin/1e6, label='_nolegend_', linestyle='dotted',color=cols[k])
                axs[j].plot(t, ymax/1e6, label='_nolegend_', linestyle='dotted',color=cols[k])
                axs[j].fill_between(t,ymin/1e6,ymax/1e6,color=cols[k], alpha=0.2)
                maximum = max(abs(ymax[-1])/1e6, abs(ymin[-1])/1e6)
            else:
                axs[j].plot(t, y, color=cols[k])
                axs[j].plot(t, ymin, label='_nolegend_', linestyle='dotted',color=cols[k])
                axs[j].plot(t, ymax, label='_nolegend_', linestyle='dotted',color=cols[k])
                axs[j].fill_between(t,ymin,ymax,color=cols[k], alpha=0.2)               
                maximum = max(abs(ymax[-1]), abs(ymin[-1]))
 
            # Subplot title
            axs[j].set_title(group_names[j])


            # Finding top point of graph
            if len(maxval) < j + 1:
                maxval.append(maximum)
            elif maximum > maxval[j]:
                maxval[j] = maximum

            # Shrinking and moving plot to better position
            box = axs[j].get_position()
            axs[j].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            
            axs[j].set_xlabel("Time [s]")
            
            # Adding semi-transparent grid
            axs[j].grid(alpha=0.7)
        
        # Ensuring proper axis limits
        for ax in axs:
            ax.set_ylim(bottom=0, top = 1.1*abs(max(maxval[0], maxval[1])))
            ax.plot(600*np.ones(200), np.linspace(0,1.5*(abs(maximum)),200), linestyle='dashed', color="blue", alpha=0.7, linewidth=0.8)
            ax.set_xlim([0,2500])

        # Removing tick labels for rightmost subplot
        axs[1].axes.yaxis.set_ticklabels([])
        #axs[1].legend(legend, loc='upper center', bbox_to_anchor=(-0.03, -0.14),fancybox=True, shadow=True, ncol=3)
        
        # Subplots share y axis, only leftmost need ylabel
        if com < 3:
            axs[0].set_ylabel("Velocity [mm/s]")
        else:
            axs[0].set_ylabel("Velocity [nm/s]")
        plt.savefig(f"figs/avg_vel_{comps[com]}_{region}.pdf", bbox_inches='tight')
        plt.show()

    