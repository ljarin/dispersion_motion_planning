from motion_primitives_py import MotionPrimitiveGraph
import motion_primitives_py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import ujson as json
import os
from multiprocessing import Pool
from copy import deepcopy


class MotionPrimitiveLattice(MotionPrimitiveGraph):
    """
    A class that provides functions to compute a lattice of minimum dispersion
    points in the state space connected by feasible trajectories
    """
    @classmethod
    def load(cls, filename, plot=False):
        """
        create a motion primitive lattice from a given json file
        """
        # read from JSON file
        with open(filename) as json_file:
            data = json.load(json_file)
            print("Reading lattice from", filename, "...")

        # build motion primitive lattice from data
        mpl = cls(control_space_q=data["control_space_q"],
                  num_dims=data["num_dims"],
                  max_state=data["max_state"],
                  motion_primitive_type=getattr(motion_primitives_py, data["mp_type"]),
                  tiling=data["tiling"], plot=plot)
        mpl.dispersion = data["dispersion"]
        mpl.dispersion_list = data["dispersion_list"]
        mpl.mp_subclass_specific_data['rho'] = data["rho"]
        mpl.num_dense_samples = np.array(data["num_dense_samples"])
        mpl.vertices = np.array(data["vertices"])
        mpl.edges = np.empty((len(mpl.vertices)*mpl.num_tiles, len(mpl.vertices)), dtype=object)
        if data.get("poly_order", 0) > 0:
            mpl.poly_order = np.array(data["poly_order"])
            mpl.mp_subclass_specific_data['dynamics'] = mpl.motion_primitive_type.get_dynamics_polynomials(mpl.poly_order)

        for i in range(len(mpl.edges)):
            for j in range(len(mpl.vertices)):
                mpl.edges[i, j] = mpl.motion_primitive_type.from_dict(
                    data["edges"][i * len(mpl.vertices) + j], mpl.num_dims,
                    mpl.max_state, mpl.mp_subclass_specific_data)
        print("Lattice successfully read")
        return mpl

    def save(self, filename=None):
        """
        save the motion primitive lattice to a JSON file
        """
        # convert the motion primitives to a form that can be written
        mps = []
        for i in range(len(self.edges)):
            for j in range(len(self.vertices)):
                mp = self.edges[i, j]
                if mp != None:
                    mps.append(mp.to_dict())
                else:
                    mps.append({})
        # write the JSON file
        # print(mps)
        if filename is None:
            filename = self.saving_file_prefix
        self.dispersion_list = [x for x in self.dispersion_list if x != np.inf]
        if self.dispersion == np.inf:
            self.dispersion = 0

        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(filename, "w") as output_file:
            print("Saving lattice to", filename, "...")
            saved_params = {"mp_type": self.motion_primitive_type.__name__,
                            "control_space_q": self.control_space_q,
                            "num_dims": self.num_dims,
                            "tiling": True if self.num_tiles > 1 else False,
                            "rho": self.mp_subclass_specific_data.get('rho', 1),
                            "poly_order": self.mp_subclass_specific_data.get('poly_order', 0),
                            "dispersion": self.dispersion,
                            "dispersion_list": self.dispersion_list,
                            "check_backwards_dispersion": self.check_backwards_dispersion,
                            "max_state": self.max_state.tolist(),
                            "num_dense_samples": self.num_dense_samples,
                            "vertices": self.vertices.tolist(),
                            "edges": mps,
                            }

            json.dump(saved_params, output_file, indent=4)
            print("Lattice successfully saved")

    def compute_mp(self, inputs):
        """
        Given start and end points, compute the motion primitive
        (a wrapper for MotionPrimitive that passes in MotionPrimitiveGraph parameters)
        """
        start_pts = inputs[0]
        end_pts = inputs[1]
        mp = self.motion_primitive_type(start_pts, end_pts,
                                        self.num_dims, self.max_state, mp_subclass_specific_data)
        mp.subclass_specific_data = {}  # hacky stuff to avoid pickling lambda functions
        if not mp.is_valid:
            mp.cost = np.nan
        return mp

    def multiprocessing_init(self):
        # hacky stuff to avoid pickling lambda functions
        global mp_subclass_specific_data
        mp_subclass_specific_data = self.mp_subclass_specific_data

    def multiprocessing_dispersion_distance_fn_trajectory(self, pool, start_pts, end_pts):
        """
        A function that evaluates the cost of a path from an array of start_pts
        to an array of end_pts. The cost is the time of the optimal path.
        Uses the python multiprocessing library to use multiple cores of the computer.
        The multiprocessing pool is passed in since it only needs to be created once.
        """
        start_and_end_pts = list(itertools.product(start_pts, end_pts))
        dynamics = None
        if self.mp_subclass_specific_data.get('dynamics', None) != None:
            dynamics = deepcopy(self.mp_subclass_specific_data['dynamics'])
            self.mp_subclass_specific_data['dynamics'] = None  # hacky stuff to avoid pickling lambda functions
        pool_output = pool.map(self.compute_mp, start_and_end_pts)
        min_score = np.array([mp.cost for mp in pool_output]).reshape(start_pts.shape[0], end_pts.shape[0])
        mp_list = np.array(pool_output).reshape(start_pts.shape[0], end_pts.shape[0])
        self.mp_subclass_specific_data['dynamics'] = dynamics
        return min_score, mp_list

    def dispersion_distance_fn_trajectory(self, start_pts, end_pts):
        """
        A function that evaluates the cost of a path from an array of start_pts
        to an array of end_pts. The cost is the time of the optimal path.
        """
        score = np.ones((len(start_pts), len(end_pts))) * np.nan
        mp_list = np.empty((len(start_pts), len(end_pts)), dtype=object)
        for i in range(len(start_pts)):
            for j in range(len(end_pts)):
                mp = self.motion_primitive_type(start_pts[i, :], end_pts[j, :],
                                                self.num_dims, self.max_state,
                                                self.mp_subclass_specific_data)
                if mp.is_valid:
                    mp_list[i, j] = mp
                    score[i, j] = mp.cost
        return score, mp_list

    def compute_min_dispersion_points(self, num_output_pts, potential_sample_pts, check_backwards_dispersion=False, dispersion_threshhold=None, animate=False):
        """
        Computes the minimum dispersion set of points iteratively
        Inputs:
            num_output_pts, number of minimum dispersion vertices in the output
            potential_sample_pts, the candidates to downsample from (often generate from uniform_state_set)
            check_backwards_dispersion, whether to compute dispersion only from the dense sampling (potential_sample_pts) to the downsampled vertices, or whether to also compute the cost of the reverse motion primitive (maybe required for proof of completeness for non-symmetric systems)
            animate, whether to make an animation of the iterations
        Outputs:
            vertices, the nodes of the "minimum dispersion graph"
            edges, the connections of the graph
        """
        # overloaded from motion_primitive_graph for the moment
        # TODO maybe unify with original version used in tree

        # Take the all zero state as the first sample. If it does not exist, create it.
        origin_idx = np.flatnonzero(np.all(potential_sample_pts == 0, axis=1))
        if len(origin_idx) > 0:
            index = origin_idx[0]
        else:
            index = 0
            potential_sample_pts = np.vstack((np.zeros(self.n), potential_sample_pts))

        # initialize data structures
        mp_adjacency_matrix_fwd = np.empty((num_output_pts * self.num_tiles, len(potential_sample_pts)), dtype=object)
        actual_sample_indices = np.zeros((num_output_pts)).astype(int)
        min_score = np.ones((len(potential_sample_pts), 2)) * np.inf
        # create multiprocessing pool to compute MPs
        pool = Pool(initializer=self.multiprocessing_init)

        # each time through loop add point to the set and update data structures
        print("potential sample points:", len(potential_sample_pts))
        for i in range(num_output_pts):

            # add index to the list of sample node indices
            actual_sample_indices[i] = np.array((index))
            print(potential_sample_pts[index])

            # update scores of nodes
            min_score[index, 0] = -np.inf  # give node we chose low score
            if self.num_tiles > 1:
                end_pts = self.tile_points([potential_sample_pts[index, :]])
            else:
                end_pts = potential_sample_pts[index, :][np.newaxis, :]

            min_score_fwd, mp_list_fwd = self.multiprocessing_dispersion_distance_fn_trajectory(pool, potential_sample_pts, end_pts)
            if check_backwards_dispersion:
                min_score_bwd, _ = self.multiprocessing_dispersion_distance_fn_trajectory(pool, end_pts, potential_sample_pts)
                # min_score[:, 1] = np.nanmin(np.maximum(min_score_fwd, min_score_bwd.T),axis=1)
                min_score[:, 1] = np.maximum(np.nanmin(min_score_fwd, axis=1), np.nanmin(min_score_bwd.T, axis=1))
            else:
                min_score[:, 1] = np.nanmin(min_score_fwd, axis=1)

            min_score[:, 0] = np.nanmin(min_score, axis=1)

            if np.isnan(np.max(min_score[:, 0])):
                if i == 1:
                    print("ERROR: some sample points have no valid trajectories from the origin. Exiting")
                else:
                    print("ERROR: I don't expect this :( terrible things")
                raise SystemExit
            else:

                index = np.squeeze(np.argwhere(min_score[:, 0] == np.max(min_score[:, 0])), axis=1)
                # Do tie-breaking with picking node furthest from other nodes in sample set in state space
                if index.shape[0] > 1:
                    dispersion_simple_norm = self.dispersion_distance_fn_simple_norm(
                        potential_sample_pts[index], potential_sample_pts[actual_sample_indices[:i+1]])[0]
                    index = index[np.argmax(np.linalg.norm(dispersion_simple_norm, axis=1))].item()
                else:
                    index = index[0]

            if np.isinf(min_score[index, 0]):
                print("""WARNING: no new valid trajectories to *a* point in the
                    sample set. Not exiting.""")
                # raise SystemExit

            # save motion primitives in the adjacency matrix
            mp_adjacency_matrix_fwd[i * self.num_tiles:(i + 1) * self.num_tiles, :] = mp_list_fwd.T

            # update dispersion metric
            self.dispersion = max(min_score[:, 0])
            if np.isnan(self.dispersion):
                self.dispersion = 10**10
            self.dispersion_list.append(self.dispersion)
            # print(
            #     f"Average edges per vertex: {sum([1 for mp in np.nditer(mp_adjacency_matrix_fwd[:, actual_sample_indices], ['refs_ok']) if mp != None and mp.item().cost < 2*self.dispersion]) / len(potential_sample_pts[actual_sample_indices])}")

            print(f"MP {i + 1}/{num_output_pts}, Dispersion = {self.dispersion}")
            if dispersion_threshhold == -1 and self.dispersion != np.inf:
                copy = deepcopy(self)
                asi_copy = deepcopy(actual_sample_indices)
                asi_copy = asi_copy[:i+1]
                copy.vertices = potential_sample_pts[asi_copy]
                copy.edges = mp_adjacency_matrix_fwd[:len(asi_copy)*self.num_tiles, asi_copy]
                copy.limit_connections(2*copy.dispersion)
                disp_list = [i for i in self.dispersion_list if i != np.inf]
                if self.dispersion < 10e5:
                    print(f"Saving at dispersion {self.dispersion:.2f}")
                    copy.save(f"{self.saving_file_prefix}{len(disp_list)}.json")
                del copy

            elif dispersion_threshhold is not None and self.dispersion < dispersion_threshhold:
                print(f"Reached Dispersion Threshhold {dispersion_threshhold}")
                # Remove extra unused actual sample indices at the back
                actual_sample_indices = actual_sample_indices[:i+1]
                break

        pool.close()  # end multiprocessing pool

        # create graph representation to return
        vertices = potential_sample_pts[actual_sample_indices]
        edges = mp_adjacency_matrix_fwd[:, actual_sample_indices]

        # create an animation of the dispersion set growth
        if animate:
            self.make_animation_min_dispersion_points(actual_sample_indices,
                                                      mp_adjacency_matrix_fwd,
                                                      vertices,
                                                      potential_sample_pts)
        return vertices, edges

    def compute_min_dispersion_space(self, num_output_pts, num_dense_samples, check_backwards_dispersion=False, animate=False, dispersion_threshhold=None):
        """
        Using the bounds on the state space, compute a set of minimum dispersion
        points (similar to original Dispertio paper) and save the resulting
        graph as a class attribute

        Input:
            num_output_pts, desired number of samples (M) in the set
        """
        self.num_dense_samples = num_dense_samples
        self.check_backwards_dispersion = check_backwards_dispersion

        potential_sample_pts = self.sobol_state_sampling(self.max_state[:self.control_space_q], num_dense_samples)
        self.vertices, self.edges = self.compute_min_dispersion_points(
            num_output_pts, potential_sample_pts, check_backwards_dispersion, dispersion_threshhold, animate)
        if self.plot:
            if self.num_dims == 2:
                self.ax.plot(self.vertices[:, 0], self.vertices[:, 1], 'og')
            if self.num_dims == 3:
                self.ax_3d.plot(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 'og')

    def limit_connections(self, cost_threshold):
        """
        Examine the graph of motion primitives making up the lattice and remove
        edges that have costs greater than a given threshold

        Input:
            cost_threshold, max allowable cost for any edge in returned graph
        """
        tiled_verts = self.tile_points(self.vertices)
        if self.plot:
            self.ax.plot(self.vertices[:, 0], self.vertices[:, 1], 'og', zorder=5)
            if self.num_tiles > 1:
                self.ax.plot(tiled_verts[:, 0], tiled_verts[:, 1], 'ob', zorder=4)

        # remove any edges with cost > 2d
        for i in range(len(self.edges)):
            for j in range(len(self.vertices)):
                mp = self.edges[i, j]
                if mp is not None and mp.is_valid and mp.cost < cost_threshold + 1e-5:
                    pass
                    # if self.plot:
                    #     mp.subclass_specific_data = self.mp_subclass_specific_data
                    #     mp.plot(position_only=True, ax=self.ax)
                else:
                    self.edges[i, j] = None

        # remove any edges that end in vertices that dont have any outgoing edges
        for j in range(len(self.vertices)):
            outgoing_edges = self.edges[:, j]
            if not any(outgoing_edges):  # remove any ingoing edges if all outgoing are bad
                for i in range(1, self.num_tiles+1):
                    self.edges[j*self.num_tiles, :] = None

    def plot_config(self, ax=None, plot_mps=False):
        """
        Plot the graph and motion primitives projected into the 2D or 3D
        configuration space.
        """
        tiled_verts = self.tile_points(self.vertices)

        if ax is None:
            _, ax = plt.subplots(1, 1, subplot_kw={'projection': {2: 'rectilinear', 3: '3d'}[self.num_dims]})
        if self.num_dims == 2:
            ax.plot(self.vertices[:, 0], self.vertices[:, 1], 'og', zorder=5)
            if self.num_tiles > 1:
                ax.plot(tiled_verts[:, 0], tiled_verts[:, 1], 'o', color='palegreen', zorder=4)
        elif self.num_dims == 3:
            ax.plot(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 'og', zorder=5)
            if self.num_tiles > 1:
                ax.plot(tiled_verts[:, 0], tiled_verts[:, 1], tiled_verts[:, 2], 'o', color='palegreen', zorder=4)

        if plot_mps:
            for i in range(len(self.edges)):
                for j in range(len(self.vertices)):
                    mp = self.edges[i, j]
                    if mp != None and mp.is_valid:
                        mp.subclass_specific_data['dynamics'] = self.mp_subclass_specific_data.get('dynamics')
                        mp.plot(position_only=True, ax=ax)
        return ax

    def get_neighbor_mps(self, node_index):
        """
        return the indices and costs of nodes that are neighbors of the given
        node index

        Input:
            node_index, index of queried node in list of vertices

        Output:
            neighbors, list of tuples with entries corresponding to neighbors
                and the MotionPrimitive object representing the trajectory
                to get to them respectively
        """
        neighbors = []
        reset_map_index = int(np.floor(node_index / self.num_tiles))
        for i, mp in enumerate(self.edges[:, reset_map_index]):
            if mp is not None and mp.is_valid:
                neighbors.append(mp)
        return neighbors

    def find_mps_to_lattice(self, state):
        """
        Given an arbitrary state, return a list of motion primitives to connect
        it to the lattice

        Input:
            state, point in state space

        Output:
            connections, list of tuples with entries corresponding to node
                indices and the MotionPrimitive object respectively
        """
        # build list of neighbors
        connections = []
        for i, vertex in enumerate(self.vertices):
            mp = self.motion_primitive_type(state, vertex,
                                            self.num_dims, self.max_state,
                                            self.mp_subclass_specific_data)
            if mp.is_valid:
                connections.append((i, mp))
            # if len(connections)>2:
            #     break
        return connections

    def tile_points(self, pts):
        """
        Given a set of points in state space, return the set of points that
        copies the original set into the neighbors of an 8 or 26-connected grid.
        Each cell in this grid will have dimensions corresponding to the
        position bounds of the motion primitive lattice.

        Input:
            pts, (M, N) a set of M points each of N dimension
        Output:
            tiled_pts, (L, N) the tiled set of input points.
                L is 9M or 27M depending on the dimension of the state space
                each row that is a multiple of M corresponds to M[row]
        """
        bounds = 2 * np.array([0, -self.max_state[0], self.max_state[0]])
        tiled_pts = np.array([pts for i in range(3 ** self.num_dims)])
        if self.num_dims == 2:
            offsets = itertools.product(bounds, bounds)
        elif self.num_dims == 3:
            offsets = itertools.product(bounds, bounds, bounds)
        for i, offset in enumerate(offsets):
            tiled_pts[i, :, :self.num_dims] += offset
        return tiled_pts.reshape(len(pts) * 3 ** self.num_dims, self.n)

    def animation_helper(self, k, ax1, costs_mat, sample_inds, adj_mat, vertices, potential_sample_pts):
        print(f"frame {k+1}/{2*vertices.shape[0]}")
        if k < 0:
            return self.lines
        i = int(np.floor(k/2))

        if self.num_tiles > 1:
            tiled_vertices = self.tile_points(vertices[:i+1, :])
            self.lines[2].set_data(tiled_vertices[i+1:, 0], tiled_vertices[i+1:, 1])

        self.lines[3].set_data(vertices[:i+1 + k % 2, 0], vertices[:i+1+k % 2, 1])

        # colors = plt.cm.tab20(np.linspace(0, 1, 20))
        closest_sample_pt = np.argmin(costs_mat[:(i + 1) * self.num_tiles, ], axis=0)
        max_cost = 0
        max_cost_mp = -1
        for j in range(adj_mat.shape[1]):
            mp = adj_mat[closest_sample_pt[j], j]
            if mp is not None and mp.is_valid:
                mp.subclass_specific_data = self.mp_subclass_specific_data
                if j in sample_inds[:i+1]:  # Don't plot the trajectories between actual samples
                    sp = np.array([[], []])
                else:
                    _, sp = mp.get_sampled_position(step_size=.1)
                self.lines[0][j].set_data(sp[0, :], sp[1, :])
                # self.lines[0][j].set_color(colors[closest_sample_pt[j] % 20])
                if mp.cost > max_cost:
                    max_cost_mp = j
                    max_cost = mp.cost

        self.lines[0][max_cost_mp].set_color('red')
        self.lines[0][max_cost_mp].set_linewidth(1.9)
        self.lines[0][max_cost_mp].set_zorder(20)

        # edges = adj_mat[:(i+1)*self.num_tiles, sample_inds[:i+1]]
        # for k, edge in enumerate(np.nditer(edges, flags=['refs_ok'])):
        #     mp = edge.item()
        #     if mp is not None and mp.is_valid and mp.cost <= self.dispersion_list[i]*2:
        #         mp.subclass_specific_data = self.mp_subclass_specific_data
        #         _, sp = mp.get_sampled_position()
        #         self.lines[5][k].set_data(sp[0, :], sp[1, :])
        #     else:
        #         self.lines[5][k].set_data([], [])

        # self.lines[4].set_data(range(i+1), self.dispersion_list[:i+1])
        ax1.set_title(f'Dispersion: {self.dispersion_list[i]:.2f}')
        # for vertex in vertices[:i+1, :]:
        #     circle = plt.Circle(vertex[:self.num_dims], 2*self.dispersion_list[i]*self.max_state[1], color='b', fill=False, zorder=4)
        #     ax1.add_artist(circle)
        # plt.savefig(f'plots/frame{i}.png')
        return self.lines

    def make_animation_min_dispersion_points(self, sample_inds, adj_mat, vertices, potential_sample_pts):
        save_animation = True
        if save_animation:
            import rospkg
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('motion_primitives')
            file_name = f'{pkg_path}/motion_primitives_py/data/videos/dispersion_algorithm.mp4'

            import matplotlib
            normal_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        f, ax1 = plt.subplots(1, 1)
        # f, (ax1, ax2) = plt.subplots(1, 2)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        })

        ax1.set_aspect('equal', 'box')
        # ax2.set_aspect('equal', 'box')
        tiling_mult = self.num_tiles**(1./self.num_dims)
        ax1.set_xlim(-self.max_state[0]*1.2*tiling_mult, self.max_state[0]*1.2*tiling_mult)
        ax1.set_ylim(-self.max_state[0]*1.2*tiling_mult, self.max_state[0]*1.2*tiling_mult)
        # ax1.set_title(r"Evolution of Vertex Set $\mathcal{V}$")
        # ax2.set_xlim(0, sample_inds.shape[0])
        # ax2.set_ylim(0, self.dispersion_list[0]*1.2)
        ax1.set_title("Dispersion: ")
        max_state = self.max_state
        ax1.plot([max_state[0], max_state[0]], [-3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
        ax1.plot([-max_state[0], -max_state[0]], [-3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
        ax1.plot([-3*max_state[0], 3*max_state[0]], [-max_state[0], -max_state[0]], 'k--', zorder=7)
        ax1.plot([-3*max_state[0], 3*max_state[0]], [max_state[0], max_state[0]], 'k--', zorder=7)
        ax1.plot([-3*max_state[0], 3*max_state[0]], [3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
        ax1.plot([-3*max_state[0], 3*max_state[0]], [-3*max_state[0], -3*max_state[0]], 'k--', zorder=7)
        ax1.plot([3*max_state[0], 3*max_state[0]], [-3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
        ax1.plot([-3*max_state[0], -3*max_state[0]], [-3*max_state[0], 3*max_state[0]], 'k--', zorder=7)
        # ax2.set_xlabel(r"Number of Vertices in $\mathcal{V}$")
        f.tight_layout()

        traj_lines = []
        for j in range(adj_mat.shape[1]):
            traj_lines.append(ax1.plot([], [], linewidth=.8, color='grey')[0])
        dense_sample_pt_line, = ax1.plot([], [], 'o', markersize=1, color=('skyblue'), zorder=1)
        actual_sample_pt_line, = ax1.plot([], [], 'og', zorder=10)
        tiled_pts_line, = ax1.plot([], [], 'o', zorder=10, color='palegreen')
        # dispersion_line, = ax2.plot([], [], 'ok--')
        # dispersion_line = ax1.text(0,0,'')
        edges_lines = []
        for j in range(sample_inds.shape[0]*adj_mat.shape[0]):
            edges_lines.append(ax1.plot([], [], 'k')[0])

        self.lines = [traj_lines, dense_sample_pt_line, tiled_pts_line, actual_sample_pt_line, edges_lines]
        self.lines[1].set_data(potential_sample_pts[:, 0], potential_sample_pts[:, 1, ])

        costs_mat = np.array([getattr(obj, 'cost', np.inf) if getattr(obj, 'is_valid', False) else np.inf for index,
                              obj in np.ndenumerate(adj_mat)]).reshape(adj_mat.shape)
        ani = animation.FuncAnimation(
            f, self.animation_helper, range(-1, 2*vertices.shape[0]), interval=3000, fargs=(ax1, costs_mat, sample_inds, adj_mat, vertices, potential_sample_pts), repeat=False)

        if save_animation:
            print("Saving animation to disk")
            ani.save(file_name, dpi=800)
            print("Finished saving animation")
            matplotlib.use(normal_backend)
        else:
            plt.show()

    def compute_dispersion_from_graph(self, vertices, resolution, no_sampling_value=0, colorbar_max=None, filename="test.png", middle_mp_plot=False):
        max_state = self.max_state[:self.control_space_q]
        max_state[0] = max(vertices[:, 0])
        dense_sampling, axis_sampling = self.uniform_state_set(
            max_state, resolution[:self.control_space_q], random=False, no_sampling_value=no_sampling_value)
        print(dense_sampling.shape)
        pool = Pool(initializer=self.multiprocessing_init)
        self.vertices = None
        self.edges = None

        score, adj_mat = self.multiprocessing_dispersion_distance_fn_trajectory(pool, dense_sampling, vertices)
        # if self.check_backwards_dispersion:
        #     score_bw, adj_mat_bw = self.multiprocessing_dispersion_distance_fn_trajectory(pool, vertices, dense_sampling)
        pool.close()  # end multiprocessing pool
        costs_mat = np.array([getattr(obj, 'cost', np.inf) if getattr(obj, 'is_valid', False) else np.inf for index,
                              obj in np.ndenumerate(adj_mat)]).reshape(adj_mat.shape)
        closest_sample_pt = np.argmin(costs_mat, axis=1)
        min_score = np.nanmin(score, axis=1)
        dispersion = np.nanmax(min_score)
        if colorbar_max is None:
            colorbar_max = dispersion

        if middle_mp_plot:
            num_plots = 3
        else:
            num_plots = 2
        fig, ax = plt.subplots(1, num_plots, sharey=True, sharex=True, constrained_layout=True)

        ax[1].set_aspect('equal', 'box', share=True)
        ax[1].use_sticky_edges = False

        # ax[0].plot(dense_sampling[:, 0], dense_sampling[:, 1], '.', color='grey')
        ax[0].plot(vertices[:, 0], vertices[:, 1], 'go')
        vel_norm = np.max(np.linalg.norm((vertices[:, 2:]), axis=1))*3
        for i in range(vertices.shape[0]):
            if self.n > 3:
                ax[0].arrow(vertices[i, 0], vertices[i, 1], vertices[i, 2]/vel_norm, vertices[i, 3]/vel_norm, color='k', head_width=.1)
            elif self.n == 3:
                ax[0].arrow(vertices[i, 0], vertices[i, 1], np.cos(vertices[i, 2])/5, np.sin(vertices[i, 2])/5, color='k', head_width=.1)

        colors = plt.cm.viridis(np.linspace(0, 1, 101))
        pcm = ax[num_plots-1].pcolormesh(np.array(axis_sampling[0]), np.array(axis_sampling[1]), np.amin(costs_mat, axis=1).reshape(
            (axis_sampling[0].shape[0], axis_sampling[1].shape[0])), shading='gouraud', norm=plt.Normalize(0, colorbar_max))

        fig.colorbar(pcm, ax=ax[num_plots-1], shrink=.45)

        if middle_mp_plot:
            for j in range(adj_mat.shape[0]):
                mp = adj_mat[j, closest_sample_pt[j]]
                mp.subclass_specific_data = self.mp_subclass_specific_data
                if mp.is_valid:
                    mp.plot(position_only=True, color=colors[int(np.floor(mp.cost/colorbar_max*100))], ax=ax[1], zorder=10)
            ax[1].plot(dense_sampling[:, 0], dense_sampling[:, 1], '.', color='grey')
            ax[1].plot(vertices[:, 0], vertices[:, 1], 'go', zorder=20)
            max_disp_pt = np.squeeze(dense_sampling[np.argmax(min_score)])
            ax[2].plot(max_disp_pt[0], max_disp_pt[1], 'x')

        plt.savefig(f"{filename}.png", dpi=1200, bbox_inches='tight')
        return dispersion


if __name__ == "__main__":
    # https://pythonspeed.com/articles/python-multiprocessing/
    # prevent C++ logging from locking processes
    # set_start_method('spawn')
    from motion_primitives_py import *
    import numpy as np
    import time
    # from pycallgraph import PyCallGraph, Config
    # from pycallgraph.output import GraphvizOutput
    import rospkg

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('motion_primitives')
    pkg_path = f'{pkg_path}/motion_primitives_py/motion_primitives_py/'

    tiling = True
    plot = False
    animate = False
    check_backwards_dispersion = True
    mp_subclass_specific_data = {}

    # define parameters
    # control_space_q = 2
    # num_dims = 2
    # max_state = [3.5, 2*np.pi]
    # motion_primitive_type = ReedsSheppMotionPrimitive
    # # resolution = [.51, .5]
    # num_dense_samples = 100
    # num_output_pts = 20

    # motion_primitive_type = OptimizationMotionPrimitive
    # control_space_q = 2
    # num_dims = 2
    # max_state = [.5, 3,3,2]
    # mp_subclass_specific_data = { 'iterative_bvp_max_t': 5, 'rho': 10}
    # num_dense_samples = 10
    # num_output_pts =20

    motion_primitive_type = ETHMotionPrimitive
    control_space_q = 3
    num_dims = 2
    max_state = [.75, .5, .5]
    num_dense_samples = 1000
    num_output_pts = 10
    mp_subclass_specific_data = {'rho': .1}

    # # build lattice
    mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, tiling, False, mp_subclass_specific_data)
    mpl.saving_file_prefix = f"{pkg_path}data/lattices/testing/"
    mpl.compute_min_dispersion_space(
        num_output_pts=num_output_pts, check_backwards_dispersion=check_backwards_dispersion, animate=animate, num_dense_samples=num_dense_samples, dispersion_threshhold=-1)

    mpl = MotionPrimitiveLattice.load(f"{pkg_path}data/lattices/testing/10.json", plot)
    mpl.plot_config(plot_mps=True)

    # plot
    plt.show()


# %%
