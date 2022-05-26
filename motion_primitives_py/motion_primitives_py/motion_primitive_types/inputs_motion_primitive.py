from motion_primitives_py import MotionPrimitive
from scipy.special import factorial
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np


class InputsMotionPrimitive(MotionPrimitive):
    def __init__(self, start_state, end_state, num_dims, max_state,
                 subclass_specific_data={}):
        # Unpack and parse subclass specific data
        assert('u' in subclass_specific_data), "Must provide parameter 'u'"
        assert('dt' in subclass_specific_data), "Must provide parameter 'dt'"
        self.u = subclass_specific_data['u']

        if not 'dynamics' in subclass_specific_data:
            control_space_q = int(len(start_state) / num_dims)
            subclass_specific_data['dynamics'] = self.get_dynamics_polynomials(control_space_q,
                                                                               num_dims)

        # Initialize superclass
        super().__init__(start_state, subclass_specific_data['dynamics'](start_state, self.u, subclass_specific_data['dt']),
                         num_dims, max_state, subclass_specific_data)
        # enforce state constraints on vel, acc, ... but not position
        if (abs(self.end_state)[self.num_dims:] <= np.repeat(self.max_state[1:self.control_space_q], self.num_dims)).all():
            self.is_valid = True
        self.traj_time = subclass_specific_data['dt']
        if self.subclass_specific_data.get('rho') is None:
            self.cost = self.traj_time
        else:
            self.cost = self.traj_time*(subclass_specific_data['rho'] + np.linalg.norm(self.u)**2)

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        """
        Load a inputs representation of a motion primitive from a dictionary
        """
        mp = super().from_dict(dict, num_dims, max_state)
        if mp:
            mp.u = np.array(dict["u"])
            if "dynamics" in subclass_specific_data:
                mp.subclass_specific_data['dynamics'] = subclass_specific_data['dynamics']
            else:
                mp.subclass_specific_data['dynamics'] = mp.get_dynamics_polynomials(mp.control_space_q, mp.num_dims)
        return mp

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        dict = super().to_dict()
        if dict:
            dict["u"] = self.u
        return dict

    def get_state(self, t):
        """
        Evaluate full state of a trajectory at a given time
        Input:
            t, numpy array of times to sample at
        Return:
            state, a numpy array of size (num_dims x control_space_q, len(t))
        """
        return np.array(self.subclass_specific_data['dynamics'](self.start_state, self.u, t))

    def get_sampled_states(self, step_size=0.1):
        st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
        states = self.get_state(st)
        return np.vstack((st,states))

    def get_sampled_position(self, step_size=0.1):
        st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
        states = self.get_state(st)
        sp = states[:self.num_dims, :]
        return st, sp

    def get_sampled_input(self, step_size=0.1):
        st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time / step_size) + 1))
        su = np.repeat(self.u.reshape((len(self.u), 1)), len(st), axis=1)
        return st, su

    @staticmethod
    def get_dynamics_polynomials(control_space_q, num_dims):
        start_pt = sym.Matrix([sym.symbols(f'start_pt{i}') for i in range(control_space_q * num_dims)])
        u = sym.Matrix([sym.symbols(f'u{i}') for i in range(num_dims)])
        t = sym.symbols('t')
        x = u * t**control_space_q / factorial(control_space_q)
        for i in range(control_space_q):
            x += sym.Matrix(start_pt[i * num_dims:(i + 1) * num_dims]) * t**i / factorial(i)
        pos = x
        for j in range(1, control_space_q):
            d = sym.diff(pos, t, j)
            x = np.vstack((x, d))
        x = x.T[0]
        return sym.lambdify([start_pt, u, t], x)


if __name__ == "__main__":
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup problem
    start_state = np.zeros((num_dims * control_space_q,))
    end_state = np.random.rand(num_dims * control_space_q,)
    # max_state = 100 * np.ones((num_dims * control_space_q,))
    max_state = [1, 2, 3, 4, 5]
    u = np.array([1, -1])
    dt = 1

    # polynomial
    mp = InputsMotionPrimitive(start_state, end_state, num_dims, max_state,
                               subclass_specific_data={"u": u, "dt": dt})

    # save
    assert(mp.is_valid)
    dictionary = mp.to_dict()

    # reconstruct
    mp = InputsMotionPrimitive.from_dict(dictionary, num_dims, max_state)

    # plot
    sampling_array = mp.get_sampled_states()
    mp.plot_from_sampled_states(sampling_array)
    print(mp.get_sampled_input())
    plt.show()
