from motion_primitives_py import MotionPrimitive, PolynomialMotionPrimitive
import numpy as np
import cvxpy as cvx
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


class OptimizationMotionPrimitive(PolynomialMotionPrimitive):
    """
    Find the optimal motion primitive for an n-integrator LTI system, with cost  $\sum_{t} {(||u||^2 + \rho) * t}$, given state and input constraints.
    """

    def __init__(self, start_state, end_state, num_dims, max_state,
                 subclass_specific_data={}):
        MotionPrimitive.__init__(self, start_state, end_state, num_dims, max_state,
                                 subclass_specific_data)
        self.rho = self.subclass_specific_data.get('rho', 1)  # multiplier on time in cost function

        max_t = float(self.subclass_specific_data.get('iterative_bvp_max_t', 3))
        self.steps = self.subclass_specific_data.get('iterative_bvp_steps', 4)  # number of time steps in inner_bvp
        self.max_dt = max_t/self.steps
        self.c_A, self.c_B = OptimizationMotionPrimitive.A_and_B_matrices_quadrotor(self.num_dims, self.control_space_q)

        self.max_state[0] = np.inf  # do not enforce position constraints
        self.x_box = np.repeat(self.max_state[:self.control_space_q], self.num_dims)
        self.u_box = np.repeat(self.max_state[self.control_space_q], self.num_dims)

        self.is_valid = False
        self.poly_coeffs = None
        self.poly_multiplier = None
        self.traj_time = 0
        self.num_inner_bvp_failures = 0
        self.setup_bvp()
        self.outer_bvp()
        if self.is_valid:
            self.poly_order = self.poly_coeffs.shape[1]-1

    @staticmethod
    def A_and_B_matrices_quadrotor(num_dims, control_space_q):
        """
        Generate constant A, B matrices for continuous integrator of order control_space_q
        in configuration dimension num_dims.
        """
        n = num_dims*control_space_q
        A = np.zeros((n, n))
        B = np.zeros((n, num_dims))
        for p in range(1, control_space_q):
            A[(p-1)*num_dims: (p-1)*num_dims+num_dims, p*num_dims: p*num_dims+num_dims] = np.eye(num_dims)
        B[-num_dims:, :] = np.eye(num_dims)
        return A, B

    def outer_bvp(self):
        """
        Given a function inner_bvp that finds an optimal motion primitive given a time allocation, finds the optimal time allocation.
        """
        # exponential search to find *a* feasible dt
        dt_start = 1e-2
        while self.inner_bvp(dt_start) == np.inf:
            dt_start *= 10
        # binary search to find a better minimum starting bound on dt
        begin = dt_start/10
        end = dt_start
        tolerance = 1e-2
        while (end-begin)/2 > tolerance:
            mid = (begin + end)/2
            if self.inner_bvp(mid) == np.inf:
                begin = mid
            else:
                end = mid
        dt_start = end

        # optimization problem with lower bound on dt (since below this it is infeasible). Upper bound is arbitrary
        if dt_start + 1e-4 > self.max_dt:
            # print(f"first feasible dt too high {dt_start} {self.max_dt}")
            return
        sol = minimize_scalar(self.inner_bvp, bounds=[dt_start, self.max_dt], method='bounded', options={'xatol': 1e-01, 'maxiter': 10})
        self.optimal_dt = sol.x
        self.is_valid = sol.success
        if not self.is_valid:
            pass
            # print(f"Did not find solution to outer BVP, {self.start_state},{self.end_state}")
        else:
            self.cost, self.traj, self.inputs = self.inner_bvp(self.optimal_dt, return_traj=True)
            if self.traj is None or self.cost == np.inf:
                self.is_valid = False
                # print(f"Did not find solution to outer BVP, {self.start_state},{self.end_state}")
            else:
                self.traj_time = self.optimal_dt * self.steps
                time_vec = np.linspace(0, self.optimal_dt*(self.steps), self.traj.shape[0])
                self.poly_coeffs = np.polyfit(time_vec, self.traj[:, :self.num_dims], self.n).T  # TODO what degree polynomial

    def setup_bvp(self):
        self.state_variables = []
        self.state_constraints = []
        self.input_variables = []
        self.input_constraints = []
        self.xts = []
        self.uts = []
        # obey starting condition
        x0_var = cvx.Variable(self.start_state.shape)
        self.state_constraints.append(x0_var == self.start_state)
        self.state_variables.append(x0_var)
        R = np.eye(self.num_dims)
        for _ in range(self.steps):
            xt = cvx.Variable(self.start_state.shape)
            ut = cvx.Variable(R.shape[-1])
            self.xts.append(xt)
            self.uts.append(ut)

            # make this obey dynamics and box constraints
            self.state_constraints += [xt >= -self.x_box, xt <= self.x_box]
            self.input_constraints += [ut >= -self.u_box, ut <= self.u_box]

            # add these to state variables and input variables so we can extract them later
            self.state_variables.append(xt)
            self.input_variables.append(ut)

        # obey terminal condition
        self.state_constraints.append(self.state_variables[-1] == self.end_state)

    def inner_bvp(self, dt, return_traj=False):
        """
        Computes an optimal MP between a start and goal state, under bounding box constraints with a given time interval (dt) allocation.
        Accomplishes this by constraining x(t) and u(t) at discrete steps to obey the input, state, and dynamic constraints.
        Note that it is parameterized by time step size (dt), with a fixed number of time steps set in the constructor.
        """
        if self.num_inner_bvp_failures > 20:
            return

        # Transform a continuous to a discrete state-space system, given dt
        sysd = cont2discrete((self.c_A, self.c_B, np.eye(self.n), 0), dt)
        A = sysd[0]
        B = sysd[1]
        cost = 0  # initializing cost

        dynamic_constraints = []
        R = np.eye(self.num_dims)
        for i in range(len(self.xts)):
            # make this obey dynamics and box constraints
            dynamic_constraints.append(self.xts[i] == A @ self.state_variables[i] + B @ self.uts[i])
            cost += cvx.quad_form(self.uts[i], R*dt) + self.rho*dt  # $\sum_{t} {(||u||^2 + \rho) * t}$

        objective = cvx.Minimize(cost)
        constraints = self.state_constraints + dynamic_constraints + self.input_constraints
        prob = cvx.Problem(objective, constraints)
        try:
            total_cost = prob.solve()
        except:
            self.num_inner_bvp_failures += 1
            total_cost = np.inf
            # print(f"inner bvp failure {self.start_state}, {self.end_state}")
        # print("Solution is {}".format(prob.status))
        if return_traj:
            trajectory = None
            inputs = None
            try:
                trajectory = np.array([state.value for state in self.state_variables])
                inputs = np.array([control.value for control in self.input_variables])
            except:
                pass
            #     print("No trajectory to return")
            return total_cost, trajectory, inputs
        else:
            return total_cost

    def plot_inner_bvp_sweep_t(self):
        plt.figure()
        data = []
        for t in np.arange(.1, 10, .6):
            cost = self.inner_bvp(t/(self.steps))
            data.append((t, cost))
        data = np.array(data)
        plt.plot(data[:, 0], data[:, 1], 'bo', label="Feasible inner_bvp solutions")
        infeasible = data[data[:, 1] == np.inf, 0]
        plt.plot(infeasible, np.zeros_like(infeasible), 'ro', label="Infeasible inner_bvp at these times")
        plt.xlabel("Trajectory Time")
        plt.ylabel(r"Cost $\sum_{t} {(||u||^2 + \rho) * t}$")
        if self.is_valid:
            plt.plot(self.optimal_dt*self.steps, self.cost, '*m', markersize=10, label="outer_bvp optimum")
        plt.legend()

    def plot_outer_bvp_x(self):
        if self.is_valid:
            fig, ax = plt.subplots(self.n + self.num_dims, sharex=True)
            time_vec = np.linspace(0, self.optimal_dt*(self.steps), self.traj.shape[0])
            for i in range(self.n):
                ax[i].plot(time_vec, self.traj[:, i], label="trajectory")
                ax[i].plot(0, self.start_state[i], 'og', label="goal")
                ax[i].plot(time_vec[-1], self.end_state[i], 'or', label="start")
                ax[i].set_ylabel(fr"$x_{i}$")
                st = np.linspace(0, self.optimal_dt*(self.steps), 100)
                deriv_num = int(np.floor(i/self.num_dims))
                p = self.evaluate_polynomial_at_derivative(deriv_num, st)[i % self.num_dims]
                thresholded = np.maximum(np.minimum(p, self.max_state[deriv_num]), -self.max_state[deriv_num])
                ax[i].plot(st, thresholded)
            time_vec = np.linspace(0, self.optimal_dt*(self.steps-1), self.inputs.shape[0])
            for i in range(self.num_dims):
                ax[self.n+i].plot(time_vec, self.inputs[:, i])
                ax[self.n + i].set_ylabel(fr"$u_{i}$")
                st = np.linspace(0, self.optimal_dt*(self.steps-1), 100)
                p = self.evaluate_polynomial_at_derivative(self.control_space_q, st)[i % self.num_dims]
                thresholded = np.maximum(np.minimum(p, self.max_state[deriv_num]), -self.max_state[deriv_num])
                ax[self.n+i].plot(st, thresholded)

        ax[-1].set_xlabel("Trajectory time [s]")

    def plot_outer_bvp_sweep_rho(self):
        self.rho = .01
        fig, ax = plt.subplots(self.n + self.num_dims)
        for _ in range(5):
            self.rho *= 10
            self.outer_bvp()
            if self.is_valid:
                time_vec = np.linspace(0, self.optimal_dt*(self.steps-1), self.inputs.shape[0])
                for i in range(self.num_dims):
                    ax[i].plot(time_vec, self.inputs[:, i], label=rf"$\rho =$ {self.rho}")
                    ax[i].set_ylabel(fr"$u_{i}$")
                time_vec = np.linspace(0, self.optimal_dt*(self.steps), self.traj.shape[0])
                for i in range(self.n):
                    ax[i + self.num_dims].plot(time_vec, self.traj[:, i], label=rf"$\rho =$ {self.rho}")
                    ax[i + self.num_dims].set_ylabel(fr"$x_{i}$")
        ax[-1].set_xlabel("Trajectory time [s]")
        plt.legend(loc="center right")


if __name__ == "__main__":
    import time

    from pycallgraph import PyCallGraph, Config
    from pycallgraph.output import GraphvizOutput
    rho = 1e3
    max_t = 20
    num_dims = 2

    start_state = np.array([0.63703125, 0.63703125, 1.9,1.9, 1,1])  # initial state
    end_state = np.array([0.,  0, 0.,  0., 0, 0])  # terminal state
    # end_state = np.zeros(num_dims*control_space_q)  # terminal state
    max_state = [0, 2, 3, 10]
    subclass_specific_data = {'rho': rho, 'iterative_bvp_max_t': max_t, 'iterative_bvp_steps' : 10}

    with PyCallGraph(output=GraphvizOutput(), config=Config(max_depth=5)):

        mp = OptimizationMotionPrimitive(start_state, end_state, num_dims, max_state,
                                         subclass_specific_data=subclass_specific_data)
    t = time.time()

    mp = OptimizationMotionPrimitive(start_state, end_state, num_dims, max_state,
                                     subclass_specific_data=subclass_specific_data)
    elapsed = time.time() - t
    print(elapsed)
    print(mp.traj_time)

    # mp.plot_inner_bvp_sweep_t()
    # mp.plot_outer_bvp_x()
    # mp.plot_outer_bvp_sweep_rho()

    # dictionary = mp.to_dict()
    # mp = OptimizationMotionPrimitive.from_dict(dictionary, num_dims, max_state)

    mp.plot()
    plt.show()
