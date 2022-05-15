import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

def clean_transfer_data(region, comp, comp2, group):

    with open(f"txtfiles/avg_transfer_{region}_{comp}_{comp2}_{group}.txt", 'r') as infile:
        lines = infile.readlines()
        filename = f"avg_transfer_{region}_{comp}_{comp2}_{group}.txt"
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
    
    line_finished = False
    while not line_finished:
        if lines[-1][0] != "t":
            lines[-2] = lines[-2][:-1] + lines[-1] + "\n"
            lines = lines[:-1]
        if lines[-1][0] == "t":
            line_finished = True

    with open(f"txtfiles/avg_transfer_{region}_{comp}_{comp2}_{group}.txt", 'w') as outfile:
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


def clean_velocity_data(region, comp, group):

    with open(f"txtfiles/avg_vel_{region}_{comp}_{group}.txt", 'r') as infile:
        lines = infile.readlines()
        j = 0
        for i,line in enumerate(lines):
            completed = False
            k = i
            while not completed:   
                if k + 1 < len(lines):
                    if lines[i+1][0] == 'o':
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
    
    if lines[-1][0] != "o":
        lines[-2] = lines[-2][:-1] + lines[-1] + "\n"
        lines = lines[:-1]
    
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
    
    if lines[-1][-3:] == "]]\n":
        lines[-1] = lines[-1][:-2] + "\n"
    with open(f"txtfiles/avg_vel_{region}_{comp}_{group}.txt", 'w') as outfile:
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

