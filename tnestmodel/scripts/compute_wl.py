import numpy as np
from tnestmodel.datasets import datasets
from tnestmodel.temp_wl import compute_d_rounds

import time


for dataset in datasets:
    print()
    print(dataset.name)
    E = dataset.read_edges_dir()
    nodes = np.unique(E[:, :2].ravel())
    total_time_diff = np.max(E[:,2])-np.min(E[:,2])
    h2 = int(round(total_time_diff*0.2))
    print(h2)
    start = time.time()
    colors, nodes, x = compute_d_rounds(E, dataset.num_nodes, d=-1, h=h2)
    end = time.time()
    print("time", end - start)
    #print(len(colors))


