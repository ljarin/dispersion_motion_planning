from motion_primitives_py import MotionPrimitive
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.linalg import expm
import scipy.integrate as integrate


class PolynomialMotionPrimitive(MotionPrimitive):
    """
    A motion primitive constructed from polynomial coefficients
    """

    def __init__(self, start_state, end_state, num_dims, max_state,
                 subclass_specific_data={}):
        # Initialize class
        super().__init__(start_state, end_state, num_dims, max_state,
                         subclass_specific_data)
        self.poly_order = 2 * self.control_space_q - 1
        self.polynomial_setup(self.poly_order)

        # Solve boundary value problem
        self.poly_coeffs, self.traj_time = self.iteratively_solve_bvp_meam_620_style(
            self.start_state, self.end_state, self.num_dims,
            self.max_state, self.subclass_specific_data['dynamics'], subclass_specific_data.get('iterative_bvp_dt', .2), subclass_specific_data.get('iterative_bvp_max_t', 2), self.poly_multiplier)
        if self.poly_coeffs is not None:
            self.is_valid = True
            if self.subclass_specific_data.get('rho') is None:
                self.cost = self.traj_time
            else:
                self.cost = self.traj_time * self.subclass_specific_data['rho']
                st, su = self.get_sampled_input()
                self.cost += np.linalg.norm(np.sum((su)**2 * st, axis=1))

    def polynomial_setup(self, poly_order):
        self.poly_order = poly_order
        if self.subclass_specific_data.get("dynamics", None) is None:
            self.subclass_specific_data['dynamics'] = self.get_dynamics_polynomials(poly_order)
        self.poly_multiplier = np.array([np.concatenate((np.zeros(deriv_num), self.subclass_specific_data['dynamics'][deriv_num](1)))[
                                        :(poly_order+1)] for deriv_num in range(poly_order) for i in range(self.num_dims)])

    @classmethod
    def from_dict(cls, dict, num_dims, max_state, subclass_specific_data={}):
        """
        Load a polynomial representation of a motion primitive from a dictionary
        """
        mp = super().from_dict(dict, num_dims, max_state)
        if mp:
            mp.poly_multiplier = None
            mp.poly_coeffs = np.array(dict["polys"])
            if "dynamics" in subclass_specific_data:
                mp.subclass_specific_data['dynamics'] = subclass_specific_data['dynamics']
            mp.polynomial_setup(mp.poly_coeffs.shape[1]-1)
        return mp

    def to_dict(self):
        """
        Write important attributes of motion primitive to a dictionary
        """
        dict = super().to_dict()
        if dict:
            dict["polys"] = self.poly_coeffs.tolist()
        return dict

    def get_state(self, t):
        """
        Evaluate full state of a trajectory at a given time
        Input:
            t, numpy array of times to sample at
        Return:
            state, a numpy array of size (num_dims x control_space_q, len(t))
        """
        return np.vstack([self.evaluate_polynomial_at_derivative(i, [t])
                          for i in range(self.control_space_q)])

    def get_input(self, t):
        return self.evaluate_polynomial_at_derivative(self.control_space_q, t)

    def get_sampled_states(self, step_size=0.1):
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            sampled_array = np.empty((1+self.n, st.shape[0]))
            sampled_array[0, :] = st
            for i in range(self.control_space_q):
                sampled_array[1+i*self.num_dims:1+(i+1)*self.num_dims] = self.evaluate_polynomial_at_derivative(i, st)
            return sampled_array
        return None

    def get_sampled_position(self, step_size=0.1):
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time/step_size)+1))
            sp = self.evaluate_polynomial_at_derivative(0, st)
            return st, sp
        else:
            return None, None

    def get_sampled_input(self, step_size=0.1):
        if self.is_valid:
            st = np.linspace(0, self.traj_time, int(np.ceil(self.traj_time / step_size) + 1))
            su = self.get_input(st)
            return st, su
        else:
            return None, None

    def translate_start_position(self, start_pt):
        self.poly_coeffs[:, -1] = start_pt
        self.end_state[:self.num_dims] = self.end_state[:self.num_dims] - self.start_state[:self.num_dims] + start_pt
        self.start_state[:self.num_dims] = start_pt

    def evaluate_polynomial_at_derivative(self, deriv_num, st):
        """
        Sample the specified derivative number of the polynomial trajectory at
        the specified times
        Input:
            deriv_num, order of derivative, scalar
            st, numpy array of times to sample
        Output:
            sampled, array of polynomial derivative evaluated at sample times
        """
        if self.poly_multiplier is None:
            self.polynomial_setup(self.poly_order)

        return self.evaluate_polynomial_at_derivative_static(deriv_num, st, self.subclass_specific_data['dynamics'], self.poly_coeffs, self.poly_multiplier)

    @staticmethod
    def evaluate_polynomial_at_derivative_static(deriv_num, st, dynamics, polys, poly_multiplier):
        """
        Sample the specified derivative number of the polynomial trajectory at
        the specified times
        Input:
            deriv_num, order of derivative, scalar
            st, numpy array of times to sample
        Output:
            sampled, array of polynomial derivative evaluated at sample times
        """
        if (deriv_num+1)*polys.shape[0] > poly_multiplier.shape[0]:
            return None
        p = np.roll(polys, deriv_num) * poly_multiplier[deriv_num*polys.shape[0]: (deriv_num+1)*polys.shape[0], :]
        sampled = np.array([np.dot(dynamics[0](t), p.T) for t in st]).T
        return sampled

    @staticmethod
    def get_dynamics_polynomials(order):
        """
        Returns an array of lambda functions that evaluate the derivatives of
        a polynomial of specified order with coefficients all set to 1

        Example for polynomial order 5:
        time_derivatives[0] = lambda t: [t**5, t**4, t**3, t**2, t, 1]
        time_derivatives[1] = lambda t: [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0]
        time_derivatives[2] = lambda t: [20*t**3, 12*t**2, 6*t, 2, 0, 0]
        time_derivatives[3] = lambda t: [60*t**2, 24*t, 6, 0, 0, 0]

        Input:
            order, order of the polynomial
        Output:
            time_derivatives, an array of length (control_space_q + 1)
                represents the time derivatives of the specified polynomial with
                the ith element of the array representing the ith derivative
        """
        # construct polynomial of the form [T**5, T**4, T**3, T**2, T, 1]
        t = sym.symbols('t')
        x = np.squeeze(sym.Matrix([t**(order - i) for i in range(order + 1)]))

        # iterate through relevant derivatives and make function for each
        time_derivatives = []
        for _ in range(order+1):
            time_derivatives.append(sym.lambdify([t], x))
            x = sym.derive_by_array(x, t)
        return time_derivatives

    @staticmethod
    def solve_bvp_meam_620_style(start_state, end_state, num_dims, dynamics, T):
        """
        Return polynomial coefficients for a trajectory from start_state ((n,) array) to end_state ((n,) array) in time interval [0,T]
        The array of lambda functions created in get_dynamics_polynomials and the dimension of the configuration space are also required.
        """
        control_space_q = int(start_state.shape[0]/num_dims)
        poly_order = (control_space_q)*2-1
        A = np.zeros((poly_order+1, poly_order+1))
        for i in range(control_space_q):
            x = dynamics[i]  # iterate through all the derivatives
            A[i, :] = x(0)  # x(ti) = start_state
            A[control_space_q+i, :] = x(T)  # x(tf) = end_state

        polys = np.zeros((num_dims, poly_order+1))
        b = np.zeros(control_space_q*2)
        for i in range(num_dims):  # Construct a separate polynomial for each dimension

            # vector of the form [start_state,end_state,start_state_dot,end_state_dot,...]
            b[: control_space_q] = start_state[i:: num_dims]
            b[control_space_q:] = end_state[i:: num_dims]
            poly = np.linalg.solve(A, b)

            polys[i, :] = poly

        return polys

    @staticmethod
    def iteratively_solve_bvp_meam_620_style(start_state, end_states, num_dims, max_state, dynamics, dt, max_t, poly_multiplier):
        """
        Given a start and goal pt, iterate over solving the BVP until the input constraint is satisfied-ish.
        """
        def check_max_state_and_input(polys):
            critical_pts = np.zeros(polys.shape[1] + 2)
            critical_pts[:2] = [0, t]
            for k in range(1, control_space_q+1):
                u_max = 0
                for i in range(num_dims):
                    roots = np.roots((polys*dynamics[k + 1](1))[i, :])
                    roots = roots[np.isreal(roots)]
                    critical_pts[2:2+roots.shape[0]] = roots
                    critical_pts[2+roots.shape[0]:] = 0
                    critical_us = PolynomialMotionPrimitive.evaluate_polynomial_at_derivative_static(
                        k, critical_pts, dynamics, polys, poly_multiplier)
                    u_max = max(u_max, np.max(np.abs(critical_us)))
                if u_max > max_state[k]:
                    return False
            return True

        t = 0
        polys = None
        control_space_q = int(start_state.shape[0]/num_dims)
        done = False
        while not done:
            t += dt + float(np.random.rand(1)*dt/5.)
            if t > max_t:
                polys = None
                t = np.inf
                break
            polys = PolynomialMotionPrimitive.solve_bvp_meam_620_style(start_state, end_states, num_dims, dynamics, t)
            done = check_max_state_and_input(polys)
        return polys, t


if __name__ == "__main__":
    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup problem
    start_state = np.zeros((num_dims * control_space_q,))
    # end_state = np.random.rand(num_dims * control_space_q,)
    end_state = np.ones_like(start_state)*.1
    max_state = 1 * np.ones((control_space_q+1,))

    # polynomial
    mp = PolynomialMotionPrimitive(start_state, end_state, num_dims, max_state, {'rho': 1})

    # save
    assert(mp.is_valid)
    assert(np.array_equal(mp.end_state, end_state))
    print(mp.cost)
    dictionary = mp.to_dict()

    # reconstruct
    mp = PolynomialMotionPrimitive.from_dict(dictionary, num_dims, max_state)

    # plot
    mp.translate_start_position([1, 5])
    mp.plot(position_only=True)
    plt.show()
