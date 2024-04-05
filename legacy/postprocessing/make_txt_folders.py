import os

os.system("mkdir txtfiles")

#os.system("mv *txt txtfiles/")

for i in range(1,34):
    os.system(f"mkdir txtfiles/C{i}")
    os.system(f"cp patients/C{i}/plots/*.txt txtfiles/C{i}/")
    os.system(f"mmv -d 'txtfiles/C{i}/avg_vel_grey matter_*' txtfiles/C{i}/avg_vel_grey_matter_#1")
    os.system(f"mmv -d 'txtfiles/C{i}/avg_vel_white matter_*' txtfiles/C{i}/avg_vel_white_matter_#1")
    os.system(f"mmv -d 'txtfiles/C{i}/avg_vel_part of cortex_*' txtfiles/C{i}/avg_vel_cingulum_#1")
    os.system(f"mmv -d 'txtfiles/C{i}/transfer_grey matter_*' txtfiles/C{i}/transfer_grey_matter_#1")
    os.system(f"mmv -d 'txtfiles/C{i}/transfer_white matter_*' txtfiles/C{i}/transfer_white_matter_#1")
    os.system(f"mmv -d 'txtfiles/C{i}/transfer_part of cortex_*' txtfiles/C{i}/transfer_cingulum_#1")

for i in range(1,15):
    os.system(f"mkdir txtfiles/NPH{i}")
    os.system(f"cp patients/NPH{i}/plots/*.txt txtfiles/NPH{i}/")
    os.system(f"mmv -d 'txtfiles/NPH{i}/avg_vel_grey matter_*' txtfiles/NPH{i}/avg_vel_grey_matter_#1")
    os.system(f"mmv -d 'txtfiles/NPH{i}/avg_vel_white matter_*' txtfiles/NPH{i}/avg_vel_white_matter_#1")
    os.system(f"mmv -d 'txtfiles/NPH{i}/avg_vel_part of cortex_*' txtfiles/NPH{i}/avg_vel_cingulum_#1")
    os.system(f"mmv -d 'txtfiles/NPH{i}/transfer_grey matter_*' txtfiles/NPH{i}/transfer_grey_matter_#1")
    os.system(f"mmv -d 'txtfiles/NPH{i}/transfer_white matter_*' txtfiles/NPH{i}/transfer_white_matter_#1")
    os.system(f"mmv -d 'txtfiles/NPH{i}/transfer_part of cortex_*' txtfiles/NPH{i}/transfer_cingulum_#1")

os.system("mkdir txtfiles/plots")
os.system(f"cp results/plots/*.txt txtfiles/plots/")
os.system(f"mmv -d 'txtfiles/plots/avg_vel_grey matter_*' txtfiles/plots/avg_vel_grey_matter_#1")
os.system(f"mmv -d 'txtfiles/plots/avg_vel_white matter_*' txtfiles/plots/avg_vel_white_matter_#1")
os.system(f"mmv -d 'txtfiles/plots/avg_vel_part of cortex_*' txtfiles/plots/avg_vel_cingulum_#1")
os.system(f"mmv -d 'txtfiles/plots/transfer_grey matter_*' txtfiles/plots/transfer_grey_matter_#1")
os.system(f"mmv -d 'txtfiles/plots/transfer_white matter_*' txtfiles/plots/transfer_white_matter_#1")
os.system(f"mmv -d 'txtfiles/plots/transfer_part of cortex_*' txtfiles/plots/transfer_cingulum_#1")