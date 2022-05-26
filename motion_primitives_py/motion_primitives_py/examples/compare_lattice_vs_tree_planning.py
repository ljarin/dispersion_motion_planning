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
mpl = MotionPrimitiveLattice.load(
    f"{pkg_path}data/lattices/opt2/dispersionopt90.json")
# mpl = MotionPrimitiveLattice.load(
#     f"{pkg_path}data/lattices/opt/dispersion22.json")
# mpl = MotionPrimitiveLattice.load(
#     f"{pkg_path}data/lattices/dispersion80.json")
# mpl = MotionPrimitiveLattice.load(
#     f"{pkg_path}data/old/2_lattice_dt60.json")
print(mpl.max_state)
start_state = np.zeros((mpl.n))
goal_state = np.zeros_like(start_state)


mpt = MotionPrimitiveTree(mpl.control_space_q, mpl.num_dims,  mpl.max_state, InputsMotionPrimitive, plot=False)
mpt.mp_subclass_specific_data['dt'] = .3
mpt.mp_subclass_specific_data['num_u_per_dimension'] = 5
mpt.mp_subclass_specific_data['rho'] = mpl.mp_subclass_specific_data['rho']

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].axis('off')
ax[1].axis('off')

gs = GraphSearch.from_yaml(f'{pkg_path}data/maps/corridor.yaml', mpt, heuristic='min_time')
gs.run_graph_search()
gs.plot(ax[0])

gs = GraphSearch.from_yaml(f'{pkg_path}data/maps/corridor.yaml', mpl, heuristic='min_time', goal_tolerance=[])
mpl.mp_subclass_specific_data['iterative_bvp_dt'] = .5
mpl.mp_subclass_specific_data['iterative_bvp_max_t'] = 10
# gs.run_graph_search()
# gs.plot(ax[1])
plt.savefig(f"{pkg_path}/data/plots/compare_to_tree.png", dpi=1200, bbox_inches='tight')
plt.show()
