import numpy as np
import matplotlib.pyplot as plt
import argparse
from IPython import embed

# Leser kommmandolinjeargumenter
parser = argparse.ArgumentParser()
parser.add_argument("-case", "--patient", type=str, default="C25")
parser.add_argument("-res", "--resolution", type=int, default=16)

args = parser.parse_args()
def read_data(patient, resolution, file="pressure_average.dat"):
    with open(f"meshes/{patient}/results{resolution}/{file}", 'r') as infile:
        lines = infile.readlines()
        legends = lines[0].split()
        p_matrix = np.zeros((len(legends),len(lines[1:])))
        for i in range(len(p_matrix[0,:])):
            line = lines[i+1].split()
            for j,x in enumerate(line):
                p_matrix[j,i] = float(x)
    return legends, p_matrix

def plot_compartment(p_matrix, legends, comp, title, filename, i_start=0):
    leg = []
    try:
        for c in comp:
            plt.plot(p_matrix[0,i_start:], p_matrix[c+1,i_start:])
            leg.append(legends[c+1])  
    except:
        plt.plot(p_matrix[0,i_start:], p_matrix[comp+1,i_start:])
        leg.append(legends[comp+1])
    plt.title(title)
    plt.legend(leg)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [mmHg]")
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    legends, p = read_data(args.patient, args.resolution)
    legends1, p1 = read_data(args.patient, args.resolution, "1.txt")
    legends2, p2 = read_data(args.patient, args.resolution, "2.txt")
    legends3, p3 = read_data(args.patient, args.resolution, "3.txt")

    plot_compartment(p, legends, [0,1,2], f"Average vascular pressure in {args.patient} parenchyma", filename=f"meshes/{args.patient}/results{args.resolution}/average_vascular_pressure_{args.patient}.pdf")

    plot_compartment(p, legends, [3,4,5], f"Average perivascular pressure in {args.patient} parenchyma", filename=f"meshes/{args.patient}/results{args.resolution}/average_perivascular_pressure_{args.patient}.pdf")

    plot_compartment(p, legends, [6], f"Average extracellular pressure in {args.patient} parenchyma", filename=f"meshes/{args.patient}/results{args.resolution}/average_extracellular_pressure_{args.patient}.pdf")

    plot_compartment(p, legends, [2, 5,6], f"Average extracellular pressure in {args.patient} parenchyma", filename=f"meshes/{args.patient}/results{args.resolution}/average_extracellular_venous_pressure_{args.patient}.pdf")

    plot_compartment(p1, legends, [0,1,2], f"Average vascular pressure in {args.patient} grey matter", filename=f"meshes/{args.patient}/results{args.resolution}/average_vascular_pressure_{args.patient}_grey.pdf")

    plot_compartment(p1, legends, [3,4,5], f"Average perivascular pressure in {args.patient} grey matter", filename=f"meshes/{args.patient}/results{args.resolution}/average_perivascular_pressure_{args.patient}_grey.pdf")

    plot_compartment(p1, legends, [6], f"Average extracellular pressure in {args.patient} grey matter", filename=f"meshes/{args.patient}/results{args.resolution}/average_extracellular_pressure_{args.patient}_grey.pdf")

    plot_compartment(p1, legends, [2, 5,6], f"Average extracellular pressure in {args.patient} grey matter", filename=f"meshes/{args.patient}/results{args.resolution}/average_extracellular_venous_pressure_{args.patient}_grey.pdf")

    plot_compartment(p2, legends, [0,1,2], f"Average vascular pressure in {args.patient} white matter", filename=f"meshes/{args.patient}/results{args.resolution}/average_vascular_pressure_{args.patient}_white.pdf")

    plot_compartment(p2, legends, [3,4,5], f"Average perivascular pressure in {args.patient} white matter", filename=f"meshes/{args.patient}/results{args.resolution}/average_perivascular_pressure_{args.patient}_white.pdf")

    plot_compartment(p2, legends, [6], f"Average extracellular pressure in {args.patient} white matter", filename=f"meshes/{args.patient}/results{args.resolution}/average_extracellular_pressure_{args.patient}_white.pdf")

    plot_compartment(p2, legends, [2, 5,6], f"Average extracellular pressure in {args.patient} white matter", filename=f"meshes/{args.patient}/results{args.resolution}/average_extracellular_venous_pressure_{args.patient}_white.pdf")

    embed()