from motion_primitives_py import MotionPrimitiveGraph
import numpy as np
import matplotlib.pyplot as plt


class MotionPrimitiveTree(MotionPrimitiveGraph):
    """
    """

    def get_neighbor_mps(self, start_pt, dt, num_u_per_dimension):
        """
        Create motion primitives for a start point by taking an even sampling over the
        input space at a given dt
        i.e. old sikang method
        """
        # create evenly sampled inputs
        max_u = self.max_state[self.control_space_q]
        single_u_set = np.linspace(-max_u, max_u, num_u_per_dimension)
        u_grid = np.meshgrid(*[single_u_set for i in range(self.num_dims)])
        u_set = np.dstack(([x.flatten() for x in u_grid]))[0]

        # convert into motion primitives
        dynamics = self.mp_subclass_specific_data['dynamics']
        mps = []
        for u in u_set:
            self.mp_subclass_specific_data['u'] = u
            mp = self.motion_primitive_type(start_pt, None, self.num_dims,
                                            self.max_state,
                                            self.mp_subclass_specific_data)
            mps.append(mp)
        return mps

    def compute_all_possible_mps(self, start_pt, num_u_set, num_dts, min_dt, max_dt):
        """
        Compute a sampled reachable set from a start point, up to a max dt

        num_u_set  Number of MPs to consider at a given time
        num_dts Number of time horizons to consider between 0 and max_dt
        min_dt  Min time horizon of MP
        max_dt Max time horizon of MP

        """
        # max control input #TODO should be a vector b/c perhaps different in Z
        max_u = self.max_state[self.control_space_q]

        single_u_set = np.linspace(-max_u, max_u, num_u_set)
        dt_set = np.linspace(min_dt, max_dt, num_dts)
        u_grid = np.meshgrid(*[single_u_set for i in range(self.num_dims)])
        u_set = np.dstack(([x.flatten() for x in u_grid]))[0].T
        sample_pts = np.array(self.quad_dynamics_polynomial(start_pt, u_set[:, :, np.newaxis], dt_set[np.newaxis, :]))
        sample_pts = np.transpose(sample_pts, (2, 1, 0))

        if self.plot:
            if self.num_dims > 1:
                plt.plot(sample_pts[:, :, 0], sample_pts[:, :, 1], marker='.', color='k', linestyle='none')
                plt.plot(start_pt[0], start_pt[1], 'og')
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
            else:
                plt.plot(sample_pts[:, :, 0], np.zeros(sample_pts.shape[0:1]), marker='.', color='k', linestyle='none')
                plt.plot(start_pt[0], 0, 'og')

        return sample_pts, dt_set, u_set

    def compute_min_dispersion_set(self, start_pt, num_u_per_dimension, num_u_set, num_dts, min_dt, max_dt):
        """
        Compute a set of num_output_mps primitives (u, dt) which generate a
        minimum state dispersion within the reachable state space after one
        step.
        """
        # TODO add stopping policy?
        num_output_mps = num_u_per_dimension**self.num_dims  # number of total motion primitives

        self.dispersion_distance_fn = self.dispersion_distance_fn_simple_norm
        score = np.ones((num_dts*num_u_set**self.num_dims, num_output_mps))*np.inf
        potential_sample_pts, dt_set, u_set = self.compute_all_possible_mps(start_pt, num_u_set, num_dts, min_dt, max_dt)
        potential_sample_pts = potential_sample_pts.reshape(
            potential_sample_pts.shape[0]*potential_sample_pts.shape[1], potential_sample_pts.shape[2])
        # Take the closest motion primitive as the first choice (may want to change later)
        first_score = np.linalg.norm(potential_sample_pts-start_pt.T, axis=1)
        closest_pt = np.argmin(first_score)
        score[:, 0] = first_score

        actual_sample_pts, actual_sample_indices = self.compute_min_dispersion_points(
            num_output_mps, potential_sample_pts, score, closest_pt)

        actual_sample_indices = np.unravel_index(actual_sample_indices, (num_dts, num_u_set**self.num_dims))
        # Else compute minimum dispersion points over the whole state space (can be quite slow) (very similar to original Dispertio)
        dts = dt_set[actual_sample_indices[0]]
        us = u_set[:, actual_sample_indices[1]]

        if self.plot:
            if self.num_dims > 1:
                plt.plot(actual_sample_pts[:, 0], actual_sample_pts[:, 1], 'om')
                self.create_evenly_spaced_mps(start_pt, max_dt/2.0, num_u_per_dimension)
            else:
                plt.plot(actual_sample_pts[:, 0], np.zeros(actual_sample_pts.shape), 'om')

        return np.vstack((dts, us))

    def create_state_space_MP_lookup_table_tree(self, num_u_per_dimension, num_state_deriv_pts, num_u_set, num_dts, min_dt, max_dt):
        """
        Uniformly sample the state space, and for each sample independently
        calculate a minimum dispersion set of motion primitives.
        """

        # Numpy nonsense that could be cleaner. Generate start pts at lots of initial conditions of the derivatives.
        # TODO replace with Jimmy's cleaner uniform_sample function
        y = np.array([np.tile(np.linspace(-i, i, num_state_deriv_pts), (self.num_dims, 1))
                      for i in self.max_state[1:self.control_space_q]])  # start at 1 to skip position
        z = np.reshape(y, (y.shape[0]*y.shape[1], y.shape[2]))
        start_pts_grid = np.meshgrid(*z)
        start_pts_set = np.dstack(([x.flatten() for x in start_pts_grid]))[0].T
        start_pts_set = np.vstack((np.zeros_like(start_pts_set[:self.num_dims, :]), start_pts_set))

        prim_list = []
        for start_pt in start_pts_set.T:
            prim_list.append(self.compute_min_dispersion_set(np.reshape(start_pt, (self.n, 1)),
                                                             num_u_per_dimension, num_u_set, num_dts, min_dt, max_dt))
            print(str(len(prim_list)) + '/' + str(start_pts_set.shape[1]))
            if self.plot:
                plt.show()

        self.start_pts = start_pts_set.T
        self.motion_primitives_list = prim_list
        self.pickle_self()


def create_many_state_space_lookup_tables(max_control_space):
    """
    Make motion primitive lookup tables for different state/input spaces
    """
    max_state = [2, 2, 1, 1, 1]
    plot = False
    moprim_list = [MotionPrimitiveTree(control_space_q, num_dims, max_state, plot)
                   for control_space_q in range(2, max_control_space) for num_dims in range(2, 3)]
    for moprim in moprim_list:
        print(moprim.control_space_q, moprim.num_dims)
        moprim.create_state_space_MP_lookup_table(num_u_per_dimension=3, num_state_deriv_pts=7,
                                                  num_u_set=20, num_dts=10, min_dt=0, max_dt=.5)


if __name__ == "__main__":
    control_space_q = 2
    num_dims = 2
    num_u_per_dimension = 9
    max_state = [1, 1, 1, 100, 1, 1]
    num_state_deriv_pts = 11
    from motion_primitives_py import InputsMotionPrimitive
    mpt = MotionPrimitiveTree(control_space_q, num_dims,  max_state, InputsMotionPrimitive, plot=True)
    start_pt = np.ones((mpt.n))
    mps = mpt.create_evenly_spaced_mps(start_pt, 1, num_u_per_dimension)
    for mp in mps:
        mp.plot(position_only=True, ax=mpt.ax)

    plt.show()
