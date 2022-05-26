# %%
from motion_primitives_py import *
import numpy as np
import time
from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput

"""
Animate the evolution of the min. dispersion algorithm
"""
tiling = True
plot = False
animate = True
check_backwards_dispersion = False
mp_subclass_specific_data = {}

# %%
# define parameters
control_space_q = 2
num_dims = 2
max_state = [3.5, 2*np.pi]
motion_primitive_type = ReedsSheppMotionPrimitive
# resolution = [.51, .5]
num_dense_samples = 100

# # # %%
# motion_primitive_type = PolynomialMotionPrimitive
# control_space_q = 2
# num_dims = 2
# max_state = [3.51, 1.51, 10, 100]
# mp_subclass_specific_data = {'iterative_bvp_dt': .1, 'iterative_bvp_max_t': 5, 'rho': 10}
# num_dense_samples = 200

# %%
# build lattice
mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, tiling, False, mp_subclass_specific_data)
mpl.compute_min_dispersion_space(
    num_output_pts=10, check_backwards_dispersion=check_backwards_dispersion, animate=animate, num_dense_samples=num_dense_samples)
