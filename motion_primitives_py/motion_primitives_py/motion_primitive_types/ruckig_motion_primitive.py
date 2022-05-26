#!/usr/bin/env python3

from motion_primitives_py import MotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
from ruckig import InputParameter, OutputParameter, Ruckig, Profile
from kr_planning_msgs.msg import SplineTrajectory, Spline, Polynomial
# from rospy_message_converter import message_converter
from copy import copy

class RuckigMotionPrimitive(MotionPrimitive):

    def __init__(self, start_state, end_state, num_dims, max_state, subclass_specific_data={}):
        super().__init__(start_state, end_state, num_dims, max_state, subclass_specific_data)
        assert(self.control_space_q == 3), "This function only works for jerk input space"
        self.run_ruckig()

    def run_ruckig(self):
        inp = InputParameter(self.num_dims)
        inp.current_position = self.start_state[:self.num_dims]
        inp.current_velocity = self.start_state[self.num_dims:2 * self.num_dims]
        inp.current_acceleration = self.start_state[2*self.num_dims:3 * self.num_dims]

        inp.target_position = self.end_state[:self.num_dims]
        inp.target_velocity = self.end_state[self.num_dims:2 * self.num_dims]
        inp.target_acceleration = self.end_state[2*self.num_dims:3 * self.num_dims]

        inp.max_velocity = np.repeat(self.max_state[1], self.num_dims)
        inp.max_acceleration = np.repeat(self.max_state[2], self.num_dims)
        inp.max_jerk = np.repeat(self.max_state[3], self.num_dims)

        ruckig = Ruckig(self.num_dims, 0.05)

        out = OutputParameter(self.num_dims)
        first_output = out

        ruckig.calculate(inp, out.trajectory)

        # print(f'Calculation duration: {first_output.calculation_duration:0.1f} [us]')
        # print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')

        self.traj_time = first_output.trajectory.duration
        if self.traj_time < 1e-10:
            self.is_valid = False
        else:
            self.is_valid = True
        self.cost = self.traj_time
        # self.subclass_specific_data['ruckig_trajectory'] = first_output.trajectory
        # self.poly_coeffs = self.get_spline_traj(first_output.trajectory)
        return first_output.trajectory

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        mp = super().from_dict(dict, num_dims, max_state)
        if mp:
            mp.run_ruckig()
        return mp

    def to_dict(self):
        dict = super().to_dict()
        return dict

    def get_state(self, t, traj=None):
        if traj is None:
            traj = self.run_ruckig()
        pos, vel, acc = traj.at_time(t)
        return np.hstack((pos, vel, acc))

    def get_sampled_states(self, step_size=0.1):
        traj = self.run_ruckig()  # TODO running too much, should be able to save it. Problems arise with pickling, which is called by deepcopy in the graph search atm
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            sampled_array = np.empty((1+self.n, st.shape[0]))
            sampled_array[0, :] = st
            for i, t in enumerate(st):
                sampled_array[1:, i] = self.get_state(t, traj)
            return sampled_array
        return None

    def get_sampled_position(self, step_size=0.1):
        if self.is_valid:
            sampled_array = self.get_sampled_states(step_size)
            return sampled_array[0, :], sampled_array[1:1+self.num_dims, :]
        return None, None

    def get_sampled_input(self, step_size=0.1):
        # Warning, finite differencing to calculate jerk
        assert step_size > 0, "Error, step_size must be >0"
        if self.is_valid:
            sampled_array = self.get_sampled_states(step_size)
            if sampled_array is not None:
                jerk = np.zeros((self.num_dims, sampled_array.shape[1]))
                acceleration = sampled_array[1+self.num_dims*2:1+self.num_dims*3, :]
                for i in range(sampled_array.shape[1]-1):
                    jerk[:, i] = (acceleration[:, i+1] - acceleration[:, i])/step_size
                return sampled_array[0, :], jerk
        return None, None

    def translate_start_position(self, start_pt):
        self.end_state[:self.num_dims] = self.end_state[:self.num_dims] - self.start_state[:self.num_dims] + start_pt
        self.start_state[:self.num_dims] = start_pt
        # self.run_ruckig()

    def get_spline_traj(self, traj):
        jerk_time_array = np.array(traj.jerks_and_times)
        sj = SplineTrajectory()
        sj.dimensions = int(self.num_dims)
        start = copy(self.start_state)
        for dim in range(self.num_dims):
            spline = Spline()
            # size 7 is hard coded in ruckig library as max # of segments
            for seg in range(7):
                poly = Polynomial()
                poly.degree = 3
                poly.dt = jerk_time_array[dim*2, seg]
                if poly.dt == 0:
                    continue
                j = jerk_time_array[dim*2 + 1, seg]
                p, v, a = start[dim::self.num_dims]
                poly.coeffs = [p, v, a/2, j/6]
                start[dim::self.num_dims] = Profile.integrate(poly.dt, p, v, a, j)
                # print(poly.coeffs)
                spline.segments += 1
                spline.segs.append(poly)
                spline.t_total += poly.dt
            sj.data.append(spline)
        return sj

    # def to_dict(self):
    #     """
    #     Write important attributes of motion primitive to a dictionary
    #     """
    #     traj = self.run_ruckig()
    #     dict = super().to_dict()
    #     if dict:
    #         dict["polys"] = message_converter.convert_ros_message_to_dictionary(self.get_spline_traj(traj))
    #     return dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from copy import deepcopy

    num_dims = 2
    control_space_q = 3

    start_state = np.zeros((num_dims * control_space_q,))
    end_state = np.random.rand(num_dims * control_space_q,)
    start_state[0] = 10
    start_state[1] = -5
    max_state = 2 * np.ones((control_space_q+1))

    mp = RuckigMotionPrimitive(start_state, end_state, num_dims, max_state)
    deepcopy(mp)

    traj = mp.run_ruckig()
    mp.to_dict()
    # print(mp.start_state)

    # mp.plot(position_only=False)
    # plt.plot(start_state[0], start_state[1], 'og')
    # plt.plot(end_state[0], end_state[1], 'or')
    # plt.show()
