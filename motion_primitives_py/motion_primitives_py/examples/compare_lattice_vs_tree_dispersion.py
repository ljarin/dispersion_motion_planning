from motion_primitives_py import *
import time
import numpy as np
import matplotlib.pyplot as plt
from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput
from copy import deepcopy

"""
Compare fully expanded graphs (up to a certain depth) of lattices vs trees
"""
depth = 1
no_sampling_value = 0
resolution = [1, np.inf]
mpl = MotionPrimitiveLattice.load("data/lattices/dispersion150.json")
mpl.check_backwards_dispersion = True

colorbar_max = None
mpl_copy = deepcopy(mpl)
plt.figure()
start_state = np.zeros((mpl_copy.n))
goal_state = np.zeros_like(start_state)

om_resolution = 1
origin = [-20, -20]
dims = [40, 40]
data = np.zeros(dims)
data = data.flatten('F')
occ_map = OccupancyMap(om_resolution, origin, dims, data)
# start_state[0:3] = np.array([20, 20, 0])*resolution
goal_state[0:3] = np.array([19, 19, 0])*om_resolution

# occ_map = OccupancyMap.fromVoxelMapBag('trees_dispersion_1.1.bag')
# start_state[0:2] = [10, 6]
# goal_state[0:2] = [22, 6]

goal_tolerance = np.ones_like(start_state)*occ_map.resolution*0

# print("Random")
# resolution = [.2, np.inf]
# num_vertices = 10
# mpl.max_state[:mpl.num_dims] = mpl.max_state[:mpl.num_dims]*.8
# max_state_mult = np.repeat(mpl.max_state[:mpl.control_space_q], mpl.num_dims)[:mpl.n]
# vertices = np.random.rand(mpl.n, num_vertices).T * 2 * max_state_mult - max_state_mult
# mpl_copy.mp_subclass_specific_data['iterative_bvp_dt'] = .2
# mpl_copy.mp_subclass_specific_data['iterative_bvp_max_t'] = 100
# dispersion = mpl_copy.compute_dispersion_from_graph(
#     vertices, resolution, no_sampling_value=no_sampling_value,  colorbar_max=colorbar_max, filename="data/plots/heatmap_random")


print("Motion Primitive Tree")
mpt = MotionPrimitiveTree(mpl_copy.control_space_q, mpl_copy.num_dims,  [
                          np.inf, np.inf, mpl_copy.max_state[2]], InputsMotionPrimitive, plot=False, mp_subclass_specific_data={'num_u_per_dimension': 5, 'dt': 1})

gs = GraphSearch(mpt, occ_map, start_state[:mpl_copy.n], goal_state[:mpl.n], goal_tolerance,
                 heuristic='zero', mp_sampling_step_size=1)
plt.title('tree')

nodes = gs.expand_all_nodes(depth)
vertices = np.array([v.state for v in nodes])
print(vertices.shape)
mpl_copy.mp_subclass_specific_data['iterative_bvp_dt'] = .1
mpl_copy.mp_subclass_specific_data['iterative_bvp_max_t'] = 10
dispersion = mpl_copy.compute_dispersion_from_graph(
    vertices, resolution, no_sampling_value=no_sampling_value,  colorbar_max=colorbar_max,  filename="data/plots/heatmap_UIS", middle_mp_plot=False)

print("Motion Primitive Lattice")
mpl.plot = False
gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n], goal_tolerance,
                 heuristic='zero', mp_sampling_step_size=1)
plt.figure()
plt.title('lattice')
nodes = gs.expand_all_nodes(depth)
vertices = np.array([v.state for v in nodes])
print(vertices.shape)
mpl.mp_subclass_specific_data['iterative_bvp_dt'] = .1
mpl.mp_subclass_specific_data['iterative_bvp_max_t'] = 10
dispersion = mpl.compute_dispersion_from_graph(vertices, resolution, no_sampling_value=no_sampling_value,
                                               colorbar_max=dispersion, filename="data/plots/heatmap_lattice", middle_mp_plot=False)
# # plt.show()
