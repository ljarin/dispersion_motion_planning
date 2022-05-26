"""
Examine the change in average degree and dispersion as more samples are added.
Plot average degree vs number of samples and dispersion vs number of sample.
Was originally created to investigate and mitigate high graph degree.
"""

# Notes:
# -- Hypothesize that the graph degree should roughly stabilizes at some value
#    after a minimum number of points, and before the test points start getting exhausted.
# -- Hypothesize that sharp cliffs in dispersion/degree and the runaway increase
#    in graph degree as the test points get exhausted are artifacts of the grid
#    initialization, and go away with fuzzing.
# -- Questions to answer:
#   -- How does number of dimensions affects average degree growth?
#   -- How does random/dithered test points affect average degree growth?


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from motion_primitives_py import *

if __name__ == '__main__':

    # Euclidean Graph in 2D
    motion_primitive_type = EuclideanMotionPrimitive
    control_space_q = 1
    num_dims = 2
    max_state = 1 * np.ones(num_dims*control_space_q)
    mp_subclass_specific_data = {}
    n_test_points = 40
    tiling = False
    check_backwards_dispersion = False
    basename = f"euclidean_lattice_{num_dims}D_num_dense_samples{n_test_points}_tiling{tiling}_backward{check_backwards_dispersion}"

    # Polynomial Graph 4D
    motion_primitive_type = PolynomialMotionPrimitive
    control_space_q = 2
    num_dims = 2
    max_state = np.array([.51, 1.51, 15])  # 1 * np.ones((control_space_q+1,))
    mp_subclass_specific_data = {'iterative_bvp_dt': .05, 'iterative_bvp_max_t': 2}
    n_test_points = 40
    tiling = False
    check_backwards_dispersion = False
    basename = f"polynomial_lattice_{num_dims}D_num_dense_samples{n_test_points}_tiling{tiling}_backward{check_backwards_dispersion}"

    # # jerks Graph in 4D
    # motion_primitive_type = JerksMotionPrimitive
    # control_space_q = 3
    # num_dims = 2
    # max_state = np.array([.51, 1.51, 1.51, 100])  # 1 * np.ones((control_space_q+1,))
    # mp_subclass_specific_data = {}
    # resolution = list(.51 * np.ones(control_space_q+1))  # Not sure about this.
    # fuzz_factor = .25  # Useful values include {0 (no fuzz), 0.25 (medium fuzz)}.
    # tiling = False
    # check_backwards_dispersion = False
    # basename = f"polynomial_lattice_{num_dims}D_{resolution[0]}res_{fuzz_factor}fuzz_exhaustive_tiling_{tiling}_backward_{check_backwards_dispersion}"

    # Either load or create the specified lattice. Lattice should not have limited edges yet.
    if Path(f'data/{basename}.json').exists():
        mpl = MotionPrimitiveLattice.load(f'data/{basename}.json', plot=False)
    else:
        mpl = MotionPrimitiveLattice(
            control_space_q,
            num_dims,
            max_state,
            motion_primitive_type,
            tiling,
            False,
            mp_subclass_specific_data)
        print(f'Used {n_test_points} test points.')
        # Compute exhaustive dispersion sequence.
        mpl.compute_min_dispersion_space(
            num_output_pts=n_test_points-1,  # This -1 is needed, but shouldn't be.
            check_backwards_dispersion=check_backwards_dispersion,
            num_dense_samples=n_test_points)
        mpl.save(f'data/lattices/{basename}.json')

    print(f'Used {n_test_points} test points.')

    # Build complete cost matrix.
    costs = np.ones(mpl.edges.shape)*np.inf
    for i in range(costs.shape[0]):
        for j in range(costs.shape[1]):
            if mpl.edges[i, j] != None:
                costs[i, j] = mpl.edges[i, j].cost

    # Calculate average degree over time given changing 2*dispersion limit.
    edge_counts = np.zeros(mpl.edges.shape[1])
    for i in range(mpl.edges.shape[1]):
        edge_counts[i] = np.count_nonzero(costs[:i*mpl.num_tiles, :i] <= 2 * mpl.dispersion_list[i])
    average_degree = edge_counts / (1+np.arange(edge_counts.size))

    # Plot average degree vs number of samples.
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.plot(range(mpl.edges.shape[1]), average_degree, color='black')
    ax.set_xlabel('number samples')
    ax.set_ylabel('average degree')
    # Plot dispersion vs number of samples (on same axes).
    ax2 = ax.twinx()
    color = 'lightgrey'
    ax2.set_ylabel('dispersion', color=color)
    ax2.plot(range(mpl.edges.shape[1]), mpl.dispersion_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.savefig(f'data/plots/{basename}.png')

    # Plot the configuration with final dispersion value.
    if num_dims <= 3:
        mpl.plot_config(plot_mps=False)

    plt.show()
