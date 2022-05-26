from motion_primitives_py import *
import numpy as np
import matplotlib.pyplot as plt
import ujson as json
import rospkg

"""
Given the same target dispersion, create graphs G(V,E) for the 'no tiling' scenario and the 'tiling' scenarios. 
In the 2D case, this is '1 (no tiling) vs. 9 (tiling)' blocks to cover the same area with the graph.
"""

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('motion_primitives')
pkg_path = f'{pkg_path}/motion_primitives_py/'

animate = False
mp_subclass_specific_data = {}

# define parameters
name = 'RS'
motion_primitive_type = ReedsSheppMotionPrimitive
control_space_q = 2
num_dims = 2
max_state = [3, 2*np.pi]
num_dense_samples = 200
num_output_pts = num_dense_samples
dispersion_threshhold = 1.05
check_backwards_dispersion = False

# # # %%
# name = 'poly'
# motion_primitive_type = PolynomialMotionPrimitive
# control_space_q = 2
# num_dims = 2
# max_state = [5.51, 1.51, 15, 100]
# mp_subclass_specific_data = {'iterative_bvp_dt': .1, 'iterative_bvp_max_t': 5, 'rho': 100}
# num_dense_samples = 100
# num_output_pts = num_dense_samples
# dispersion_threshhold = 260
# check_backwards_dispersion = True


fig, ax = plt.subplots(1, 2, sharex=True)
ax[0].set_xlim(1.1*np.array([-max_state[0], max_state[0]]))
ax[0].set_aspect('equal', 'box')
ax[1].set_aspect('equal', 'box')
mpl_not_tiled = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, False, False, mp_subclass_specific_data)
mpl_not_tiled.compute_min_dispersion_space(
    num_output_pts=num_output_pts, check_backwards_dispersion=check_backwards_dispersion, animate=animate, num_dense_samples=num_dense_samples, dispersion_threshhold=dispersion_threshhold)
mpl_not_tiled.limit_connections(2*mpl_not_tiled.dispersion)
mpl_not_tiled.plot_config(plot_mps=True, ax=ax[0])

max_state[0] = max_state[0]/3
mpl_tiled = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, True, False, mp_subclass_specific_data)
mpl_tiled.compute_min_dispersion_space(
    num_output_pts=num_output_pts, check_backwards_dispersion=check_backwards_dispersion, animate=animate, num_dense_samples=num_dense_samples, dispersion_threshhold=dispersion_threshhold)
mpl_tiled.limit_connections(2*mpl_tiled.dispersion)

plt.plot([max_state[0], max_state[0]], [-3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
plt.plot([-max_state[0], -max_state[0]], [-3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
plt.plot([-3*max_state[0], 3*max_state[0]], [-max_state[0], -max_state[0]], 'k--', zorder=7)
plt.plot([-3*max_state[0], 3*max_state[0]], [max_state[0], max_state[0]], 'k--', zorder=7)
plt.plot([-3*max_state[0], 3*max_state[0]], [3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
plt.plot([-3*max_state[0], 3*max_state[0]], [-3*max_state[0], -3*max_state[0]], 'k--', zorder=7)
plt.plot([3*max_state[0], 3*max_state[0]], [-3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
plt.plot([-3*max_state[0], -3*max_state[0]], [-3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
mpl_tiled.plot_config(plot_mps=True, ax=ax[1])

filename = f"1_vs_9_{name}_nds{num_dense_samples}_dt{dispersion_threshhold}"
plt.savefig(f"{pkg_path}data/plots/{filename}.png", dpi=1200, bbox_inches='tight')

data_dict = {}
data_dict['not_tiled_num_vertices'] = len(mpl_not_tiled.vertices)
data_dict['not_tiled_num_edges'] = len([mp for mp in list(mpl_not_tiled.edges.flatten()) if mp != None])
data_dict['not_tiled_edges_per_vertex'] = len([mp for mp in list(mpl_not_tiled.edges.flatten()) if mp != None])/len(mpl_not_tiled.vertices)
data_dict['tiled_num_vertices'] = len(mpl_tiled.vertices)
data_dict['tiled_num_edges'] = len([mp for mp in list(mpl_tiled.edges.flatten()) if mp != None])
data_dict['tiled_edges_per_vertex'] = len([mp for mp in list(mpl_tiled.edges.flatten()) if mp != None])/len(mpl_tiled.vertices)


# mpl_tiled.save(f"{pkg_path}data/lattices/{filename}_tiled_lattice.json")
# mpl_not_tiled.save(f"{pkg_path}data/lattices/{filename}_untiled_lattice.json")
# with open(f"{pkg_path}data/plots/{filename}_metadata.json", "w") as output_file:
# json.dump(data_dict, output_file, indent=4)

plt.show()
