from motion_primitives_py import MotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.linalg import expm
import scipy.integrate as integrate


class EuclideanMotionPrimitive(MotionPrimitive):
    """
    A motion primitive that is just a straight line, with the norm of the distance between start and goal as the cost.
    """

    def __init__(self, start_state, end_state, num_dims, max_state,
                 subclass_specific_data={}):
        # Initialize class
        super().__init__(start_state, end_state, num_dims, max_state,
                         subclass_specific_data)
        self.is_valid = True
        self.cost = np.linalg.norm(start_state-end_state)

    def get_sampled_position(self, step_size=0.1):
        sampling_array = self.get_sampled_states(step_size)
        return sampling_array[0, :], sampling_array[1:, :]

    def get_sampled_states(self, step_size=0.1):
        sampling = self.start_state + (self.end_state - self.start_state)*np.arange(0, 1+step_size, step_size)[:, np.newaxis]
        sampling_array = np.vstack((np.arange(0, 1+step_size, step_size), sampling[:, :self.num_dims].T))
        return sampling_array

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        """
        Load a inputs representation of a motion primitive from a dictionary
        """
        return super().from_dict(dict, num_dims, max_state)

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        return super().to_dict()


if __name__ == "__main__":
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup problem
    start_state = np.zeros((num_dims * control_space_q,))
    # end_state = np.random.rand(num_dims * control_space_q,)
    end_state = np.ones_like(start_state)
    end_state[0] = 2
    max_state = 1 * np.ones((control_space_q+1,))

    # polynomial
    mp = EuclideanMotionPrimitive(start_state, end_state, num_dims, max_state)

    # save
    assert(mp.is_valid)
    assert(np.array_equal(mp.end_state, end_state))
    print(mp.cost)
    dictionary = mp.to_dict()

    # reconstruct
    mp = EuclideanMotionPrimitive.from_dict(dictionary, num_dims, max_state)

    # plot
    mp.plot(position_only=True)
    plt.show()
