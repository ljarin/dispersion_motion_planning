from motion_primitives_py import *
import time
from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput
import rospkg
import numpy as np
import matplotlib.pyplot as plt

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('motion_primitives')
pkg_path = f'{pkg_path}/motion_primitives_py/'

dispersions = [134,112,104]
fig, ax = plt.subplots(len(dispersions), 1, sharex=True)

for i,d in enumerate(dispersions):
    mpl = MotionPrimitiveLattice.load(
        f"{pkg_path}data/lattices/dispersionopt{d}.json")
    start_state = np.zeros((mpl.n))
    goal_state = np.zeros_like(start_state)

    gs = GraphSearch.from_yaml(f'{pkg_path}data/maps/corridor.yaml', mpl, heuristic='min_time', goal_tolerance=[])
    mpl.mp_subclass_specific_data['iterative_bvp_dt'] = .5
    mpl.mp_subclass_specific_data['iterative_bvp_max_t'] = 10
    gs.run_graph_search()
    gs.plot(ax[i])


plt.savefig(f"{pkg_path}/data/plots/compare_3_dispersion.png", dpi=1200, bbox_inches='tight')
plt.show()
