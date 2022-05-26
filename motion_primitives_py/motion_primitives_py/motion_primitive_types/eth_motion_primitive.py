#!/usr/bin/env python3

from motion_primitives_py import MotionPrimitive, PolynomialMotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
try:
    from mav_traj_gen import *
except:
    print("Cannot import mav_traj_gen, computing new ETHMotionPrimitive will not work, continuing")

# TODO(laura) fix so not recomputing the trajectory at graph search time (see polynomial MP setup)


class ETHMotionPrimitive(MotionPrimitive):

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)
        self.is_valid = False
        self.traj_time = 0
        self.cost = np.inf
        self.poly_coeffs = None

        seg, cost = self.calculate_trajectory()

        if self.is_valid:
            self.cost = cost
            # self.cost = self.traj_time
            # if self.subclass_specific_data.get('rho') is None:
            #     self.cost = self.traj_time
            # else:
            #     self.cost = self.traj_time * self.subclass_specific_data['rho']
            #     st, su = self.get_sampled_input()
            #     self.cost += np.linalg.norm(np.sum((su)**2 * st, axis=1))

    def calculate_trajectory(self):
        self.is_valid = False
        dimension = self.num_dims

        derivative_to_optimize = derivative_order.JERK
        # derivative_to_optimize = derivative_order.SNAP

        start = Vertex(dimension)
        end = Vertex(dimension)

        start.addConstraint(derivative_order.POSITION, self.start_state[:self.num_dims])
        start.addConstraint(derivative_order.VELOCITY, self.start_state[self.num_dims:self.num_dims*2])
        end.addConstraint(derivative_order.POSITION, self.end_state[:self.num_dims])
        end.addConstraint(derivative_order.VELOCITY, self.end_state[self.num_dims:self.num_dims*2])
        if self.control_space_q > 2:
            start.addConstraint(derivative_order.ACCELERATION, self.start_state[self.num_dims*2:self.num_dims*3])
            end.addConstraint(derivative_order.ACCELERATION, self.end_state[self.num_dims*2:self.num_dims*3])

        vertices = [start, end]
        max_v = self.max_state[1] + .5
        max_a = self.max_state[2] + .5
        segment_times = estimateSegmentTimes(vertices, max_v, max_a)
        if segment_times[0] <= 0:
            segment_times[0] = 1
        parameters = NonlinearOptimizationParameters()
        rho = self.subclass_specific_data.get('rho', None)
        if rho is not None:
            parameters.time_penalty = rho
        opt = PolynomialOptimizationNonLinear(dimension, parameters)
        opt.setupFromVertices(vertices, segment_times, derivative_to_optimize)

        opt.addMaximumMagnitudeConstraint(derivative_order.VELOCITY, max_v)
        opt.addMaximumMagnitudeConstraint(derivative_order.ACCELERATION, max_a)

        result_code = opt.optimize()
        if result_code < 0:
            return None, None
        trajectory = Trajectory()
        opt.getTrajectory(trajectory)
        self.traj_time = trajectory.get_segment_times()[0]
        seg = trajectory.get_segments()[0]
        self.is_valid = True
        self.poly_coeffs = np.array([seg.getPolynomialsRef()[i].getCoefficients(0) for i in range(self.num_dims)])
        self.poly_coeffs = np.flip(self.poly_coeffs, axis=1)
        cost = opt.getTotalCostWithSoftConstraints()
        return seg, cost

    def get_state(self, t, seg=None):
        if seg is None:
            seg, cost = self.calculate_trajectory()
        if seg is not None:
            state = np.zeros(self.n)
            for i in range(self.control_space_q):
                state[self.num_dims*i:self.num_dims*(i+1)] = seg.evaluate(t, i)
            return state

    def get_sampled_states(self, step_size=0.1):
        """
        Return an array consisting of sample times and a sampling of the trajectory for plotting 
        Will be specific to the subclass, so we raise an error if the subclass has not implemented it
        """
        seg, cost = self.calculate_trajectory()
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            sampled_array = np.zeros((1+self.n, st.shape[0]))
            sampled_array[0, :] = st
            for i, t in enumerate(st):
                sampled_array[1:, i] = self.get_state(t, seg)
            return sampled_array
        return None

    def get_sampled_position(self, step_size=0.1):
        seg, cost = self.calculate_trajectory()
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            sp = np.zeros((self.num_dims, st.shape[0]))
            for i, t in enumerate(st):
                sp[:, i] = seg.evaluate(t, 0)
            return st, sp
        return None, None

    def get_input(self, t):
        seg, cost = self.calculate_trajectory()
        if self.is_valid:
            return seg.evaluate(t, self.control_space_q)
        return None

    def get_sampled_input(self, step_size=0.1):
        if self.is_valid:
            seg, cost = self.calculate_trajectory()
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            su = np.zeros((self.num_dims, st.shape[0]))
            for i, t in enumerate(st):
                su[:, i] = seg.evaluate(t, derivative_order.SNAP)
            return st, su
        return None, None

    def translate_start_position(self, start_pt):
        self.poly_coeffs[:, -1] = start_pt
        self.end_state[:self.num_dims] = self.end_state[:self.num_dims] - self.start_state[:self.num_dims] + start_pt
        self.start_state[:self.num_dims] = start_pt

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        mp = super().from_dict(dict, num_dims, max_state, subclass_specific_data)
        if mp:
            mp.calculate_trajectory()
        return mp

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        dict = super().to_dict()
        if dict:
            dict["polys"] = self.poly_coeffs.tolist()
        return dict

    def get_dynamics_polynomials(self):
        return PolynomialMotionPrimitive.get_dynamics_polynomials


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy

    num_dims = 2
    control_space_q = 3

    start_state = [12.5, 0.4, 0.03, 0, 0, 0]
    end_state = [12.5, 0.4, 0, 0, 0, 0]
    # end_state = np.random.rand(num_dims * control_space_q,)*2
    max_state = [10, 5, 3, 5]
    subclass_data = {'rho': 10}
    mp = ETHMotionPrimitive(start_state, end_state, num_dims, max_state, subclass_specific_data=subclass_data)
    print(mp.poly_coeffs)
    mp.plot(position_only=False)
    # plt.plot(start_state[0], start_state[1], 'og')
    # plt.plot(end_state[0], end_state[1], 'or')
    plt.show()
