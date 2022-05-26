
from motion_primitives_py import MotionPrimitive
from motion_primitives_py.c_output_redirector import stdout_redirector
import io
import numpy as np
import matplotlib.pyplot as plt
from py_opt_control import min_time_bvp


class JerksMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from a sequence of constant jerks
    """

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)
        assert(self.control_space_q == 3), "This function only works for jerk input space (and maybe acceleration input space one day)"

        # start point # ugly but this is faster than using np.split
        p0, v0, a0 = self.start_state[:self.num_dims]+1e-5, self.start_state[self.num_dims:2 *
                                                                             self.num_dims]+1e-5, self.start_state[2*self.num_dims:3*self.num_dims]+1e-5
        # end point
        p1, v1, a1 = self.end_state[:self.num_dims]+1e-5, self.end_state[self.num_dims:2 *
                                                                         self.num_dims]+1e-5, self.end_state[2*self.num_dims:3*self.num_dims]+1e-5
        # state and input limits
        v_max, a_max, j_max = self.max_state[1:1+self.control_space_q] + 1e-5  # numerical hack for library seg fault
        v_min, a_min, j_min = -self.max_state[1:1+self.control_space_q] - 1e-5
        # suppress warning/error messages from C library
        # due to a bug in pytest we'll end up with a bunch of logging errors if
        # we try to log anything within an 'atexit' hook.
        # when https://github.com/pytest-dev/pytest/issues/5502 is fixed we can
        # remove this very hacky part.
        if "suppress_redirector" in subclass_specific_data:
            self.switch_times, self.jerks = min_time_bvp.min_time_bvp(p0, v0, a0, p1, v1, a1, v_min, v_max, a_min, a_max, j_min, j_max)
        else:
            with stdout_redirector(io.BytesIO()):
                self.switch_times, self.jerks = min_time_bvp.min_time_bvp(p0, v0, a0, p1, v1, a1, v_min, v_max, a_min, a_max, j_min, j_max)

        # check if trajectory is valid
        self.traj_time = np.max(self.switch_times[:, -1])
        self.is_valid = (abs(self.get_state(self.traj_time) - self.end_state) < 1e-1).all()
        if self.is_valid:
            if self.subclass_specific_data.get('rho') is None:
                self.cost = self.traj_time
            else:
                self.cost = self.traj_time * self.subclass_specific_data['rho']
                self.cost += np.linalg.norm(np.sum((self.get_sampled_input()[1])**2 * self.get_sampled_input()[0], axis=1))

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        """
        load a jerks representation of the motion primitive from a dictionary 
        """
        mp = super().from_dict(dict, num_dims, max_state)
        if mp:
            mp.switch_times = np.array(dict["switch_times"])
            mp.jerks = np.array(dict["jerks"][0])
        return mp

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        dict = super().to_dict()
        if dict:
            dict["jerks"] = self.jerks.tolist(),
            dict["switch_times"] = self.switch_times.tolist()
        return dict

    def get_state(self, t):
        """
        Evaluate full state of a trajectory at a given time
        Input:
            t, numpy array of times to sample at
        Return:
            state, a numpy array of size (num_dims x 4), ordered (p, v, a, j)
        """

        # call to optimization library to evaluate at time t
        sj, sa, sv, sp = min_time_bvp.sample(self.start_state[:self.num_dims], self.start_state[self.num_dims:2 *
                                                                                                self.num_dims], self.start_state[2*self.num_dims:3*self.num_dims], self.switch_times, self.jerks, t)
        return np.squeeze(np.concatenate([sp, sv, sa]))  # TODO concatenate may be slow because allocates new memory

    def get_sampled_states(self, step_size=0.1):
        p0, v0, a0 = np.split(self.start_state, self.control_space_q)
        st, sj, sa, sv, sp = min_time_bvp.uniformly_sample(p0, v0, a0, self.switch_times, self.jerks, dt=step_size)
        return np.vstack((st, sp, sv, sa, sj))

    def get_sampled_position(self, step_size=0.1):
        p0, v0, a0 = np.split(self.start_state, self.control_space_q)
        st, sp = min_time_bvp.uniformly_sample_position(p0, v0, a0, self.switch_times, self.jerks, dt=step_size)
        return st, sp

    def get_sampled_input(self, step_size=None):
        if step_size:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            su = np.empty((self.num_dims, len(st)))
            for dim in range(self.num_dims):
                switch_index = 1
                for i in range(len(st)):
                    while st[i] > self.switch_times[dim, switch_index]:
                        switch_index += 1
                    su[dim, i] = self.jerks[dim, switch_index - 1]
            return st, su
        else:
            return self.switch_times, self.jerks


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup problem
    start_state = np.zeros((num_dims * control_space_q,))
    end_state = np.random.rand(num_dims * control_space_q,)
    max_state = 100 * np.ones((num_dims * control_space_q,))

    # jerks
    mp = JerksMotionPrimitive(start_state, end_state, num_dims, max_state)

    # plot
    sampling_array = mp.get_sampled_states()
    mp.plot_from_sampled_states(sampling_array)
    st, su = mp.get_sampled_input(step_size=.1)
    plt.plot(st, su[0, :])
    plt.plot(st, su[1, :])
    plt.show()
