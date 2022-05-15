import os
import sys

group = sys.argv[1]
start = int(sys.argv[2])
stop  = int(sys.argv[3])

for i in range(start, stop+1):
    os.system(f"mkdir patients/{group}{i}/results32")
    os.system(f"scp larswd@saga.sigma2.no:/cluster/projects/nn8017k/Lars/experiments/meshes/{group}{i}/results32/*.txt patients/{group}{i}/results32/")
    os.system(f"scp larswd@saga.sigma2.no:/cluster/projects/nn8017k/Lars/experiments/meshes/{group}{i}/results32/*.dat patients/{group}{i}/results32/")