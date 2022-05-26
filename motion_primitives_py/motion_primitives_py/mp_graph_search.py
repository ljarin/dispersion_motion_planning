#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop, heapify
from copy import deepcopy
import matplotlib.animation as animation
from motion_primitives_py import MotionPrimitiveLattice, MotionPrimitiveTree, OccupancyMap
import yaml


class Node:
    """
    Container for node data. Nodes are sortable by the value of (f, -g).
    """

    def __init__(self, g, h, state, parent, mp, index=None, parent_index=None, graph_depth=0):
        self.f = g + h          # total-cost
        self.g = g              # cost-to-come
        self.h = h              # heuristic
        self.state = state
        self.parent = parent
        self.mp = mp
        self.index = index
        self.parent_index = parent_index
        self.graph_depth = graph_depth
        self.is_closed = False

    def __lt__(self, other):
        return (self.f, -self.g) < (other.f, -other.g)

    def __repr__(self):
        return f"Node g={self.g}, h={self.h}, state={self.state}, parent={self.parent}, is_closed={self.is_closed}, index={self.index}, parent_index={self.parent_index}"


class GraphSearch:
    """
    Uses a motion primitive lookup table stored in a json file to perform a graph search. 
    """

    def __init__(self, motion_primitive_graph, occupancy_map, start_state, goal_state, goal_tolerance=np.empty(0), mp_sampling_step_size=0.1, heuristic='min_time'):
        # Save arguments as parameters
        self.motion_primitive_graph = motion_primitive_graph
        self.map = occupancy_map
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        assert self.start_state.shape[0] == self.motion_primitive_graph.n, "Start state must have the same dimension as the motion primitive graph"
        assert self.goal_state.shape[0] == self.motion_primitive_graph.n, "Goal state must have the same dimension as the motion primitive graph"
        self.goal_tolerance = np.array(goal_tolerance)
        self.mp_sampling_step_size = mp_sampling_step_size

        # Dimensionality parameters
        self.num_dims = self.motion_primitive_graph.num_dims
        self.control_space_q = self.motion_primitive_graph.control_space_q
        self.n = self.motion_primitive_graph.n

        self.mp_list = []
        self.succeeded = False

        # Parameter used in min time heuristic from Sikang's paper
        self.rho = self.motion_primitive_graph.mp_subclass_specific_data.get('rho', 100)*1.5
        if self.rho == 0:
            self.rho == 1
        self.heuristic_dict = {
            'zero': self.zero_heuristic,
            'euclidean': self.euclidean_distance_heuristic,
            'min_time': self.min_time_heuristic,
            'bvp': self.bvp_heuristic, }
        self.heuristic = self.heuristic_dict[heuristic]

        # self.mp_start_pts_tree = spatial.KDTree(start_state)  # self.motion_primitive_graph.start_pts)

        if type(self.motion_primitive_graph) is MotionPrimitiveTree:
            self.num_u_per_dimension = self.motion_primitive_graph.mp_subclass_specific_data['num_u_per_dimension']
            self.dt = self.motion_primitive_graph.mp_subclass_specific_data['dt']
            self.num_mps = self.num_u_per_dimension**self.num_dims
            self.get_neighbor_nodes = self.get_neighbor_nodes_evenly_spaced
        elif type(self.motion_primitive_graph) is MotionPrimitiveLattice:
            self.num_mps = len(self.motion_primitive_graph.edges)
            self.get_neighbor_nodes = self.get_neighbor_nodes_lattice

    @classmethod
    def from_yaml(cls, filename, mpg, heuristic='min_time', goal_tolerance=None):
        with open(filename, 'r') as stream:
            output = yaml.load(stream, Loader=yaml.CLoader)
            di = {}
            for d in output:
                di.update(d)
        occupancy_map = OccupancyMap(di['resolution'], di['origin'], di['dim'], di['data'])
        start_state = np.zeros(mpg.n)
        start_state[:mpg.num_dims] = di['start']
        goal_state = np.zeros(mpg.n)
        goal_state[:mpg.num_dims] = di['goal']
        if goal_tolerance is None:
            goal_tolerance = np.ones(mpg.n)
        gs = cls(mpg, occupancy_map, start_state, goal_state, goal_tolerance, mp_sampling_step_size=0.1, heuristic=heuristic)
        return gs

    def zero_heuristic(self, state):
        return 0

    def min_time_heuristic(self, state):
        # sikang heuristic 1
        return self.rho * np.linalg.norm(state[:self.num_dims] - self.goal_state[:self.num_dims], ord=np.inf)/self.motion_primitive_graph.max_state[1]

    def euclidean_distance_heuristic(self, state):
        return np.linalg.norm(state[:self.num_dims] - self.goal_state[:self.num_dims])

    def bvp_heuristic(self, state):
        cost = self.motion_primitive_graph.motion_primitive_type(
            state, self.goal_state, self.motion_primitive_graph.num_dims, self.motion_primitive_graph.max_state, self.motion_primitive_graph.mp_subclass_specific_data).cost
        if cost == None:
            return np.inf
        return cost

    def build_path(self, node):
        """
        Build path from start point to goal point using the goal node's parents.
        """
        self.path_cost = 0
        while node.mp is not None:
            self.mp_list.append(node.mp)
            self.path_cost += node.mp.cost
            if node.parent is not None:
                node = self.node_dict[node.parent.tobytes()]
            else:
                break
        self.mp_list.reverse()

    def plot(self, ax=None):

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.start_state[0], self.start_state[1], 'go', zorder=5)
        ax.plot(self.goal_state[0], self.goal_state[1], 'or', zorder=5)
        neighbor_nodes_states = np.array([node.state for node in self.neighbor_nodes]).T
        if neighbor_nodes_states.size > 0:
            ax.plot(neighbor_nodes_states[0, :], neighbor_nodes_states[1, :], '.',
                    color=('.8'), zorder=2, markeredgewidth=.2, markeredgecolor='k', markersize=4)
        self.map.plot(ax=ax)
        if self.succeeded is False:
            print("Error: Cannot plot path which does not exist.")
            return
        sampled_path = np.hstack([mp.get_sampled_states()[1:1+self.num_dims, :] for mp in self.mp_list])
        path = np.vstack([mp.start_state for mp in self.mp_list]).T
        ax.plot(sampled_path[0, :], sampled_path[1, :], zorder=4)
        ax.plot(path[0, :], path[1, :], 'o', color='lightblue', zorder=4, markersize=4)

        if self.goal_tolerance.size > 1:
            ax.add_patch(plt.Circle(self.goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False, zorder=5))

        if self.succeeded:
            ax.set_xlabel(f"Dispersion: {self.motion_primitive_graph.dispersion : 0.2f}\n Path Cost: {self.path_cost : 0.2f}\n # Collision Checks: {self.num_collision_checks}")
        plt.tight_layout()

    def get_neighbor_nodes_evenly_spaced(self, node):
        neighbor_mps = self.motion_primitive_graph.get_neighbor_mps(node.state, self.dt, self.num_u_per_dimension)
        neighbors = []
        for i, mp in enumerate(neighbor_mps):
            self.num_collision_checks += 1
            if self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size):
                state = mp.end_state
                # neighbor_node = Node(mp.cost + node.g, self.heuristic(state), state, node.state, mp, graph_depth=node.graph_depth+1) # min time cost
                neighbor_node = Node(mp.cost + node.g, self.heuristic(state), state, node.state, mp, graph_depth=node.graph_depth+1)
                neighbors.append(neighbor_node)
        node.is_closed = True
        return neighbors

    def get_neighbor_nodes_lattice(self, node):
        neighbors = []
        reset_map_index = int(np.floor(node.index / self.motion_primitive_graph.num_tiles))
        for i, mp in enumerate(deepcopy(self.motion_primitive_graph.edges[:, reset_map_index])):
            if mp is not None:
                mp.translate_start_position(node.state[:self.num_dims])
                self.num_collision_checks += 1
                if self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size):
                    neighbor_node = Node(mp.cost + node.g, self.heuristic(mp.end_state), mp.end_state, node.state, mp,
                                         index=i, parent_index=node.index, graph_depth=node.graph_depth+1)
                    neighbors.append(neighbor_node)
        node.is_closed = True
        return neighbors

    def reset_graph_search(self):
        self.node_dict = {}  # A dict where key is an state and the value is a node in the queue.
        self.queue = []      # A priority queue of nodes as a heapq.
        self.neighbor_nodes = []
        self.closed_nodes = []
        self.path = None
        self.sampled_path = None
        self.path_cost = None
        self.nodes_expanded = 0
        self.mp_list = []
        self.succeeded = False
        self.num_collision_checks = 0

        if not self.map.is_free_and_valid_position(self.start_state[:self.num_dims]):
            print("start invalid")
            self.queue = None
            return
        if not self.map.is_free_and_valid_position(self.goal_state[:self.num_dims]):
            print("goal invalid")
            self.queue = None
            return

        if type(self.motion_primitive_graph) is MotionPrimitiveLattice:
            start_position_offset = np.hstack((self.start_state[:self.num_dims], np.zeros_like(self.start_state[self.num_dims:])))
            starting_neighbors = self.motion_primitive_graph.find_mps_to_lattice(deepcopy(self.start_state) - start_position_offset)
            self.start_edges = []
            for i, mp in starting_neighbors:
                mp.translate_start_position(self.start_state[:self.num_dims])
                node = Node(mp.cost, self.heuristic(mp.end_state), mp.end_state, None, mp, index=i, parent_index=None, graph_depth=0)
                self.start_edges.append(mp)
                if self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size):
                    heappush(self.queue, node)
                    self.node_dict[node.state.tobytes()] = node
        else:
            node = Node(0, self.heuristic(self.start_state), self.start_state, None, None, graph_depth=0)
            self.node_dict[node.state.tobytes()] = node
            heappush(self.queue, node)

    def run_graph_search(self):
        self.reset_graph_search()

        while self.queue:
            node = heappop(self.queue)  # While queue is not empty, pop the next smallest total cost f node
            # If node has been closed already, skip.
            if node.is_closed:
                continue
            # Otherwise, expand node and for each neighbor...
            self.nodes_expanded += 1
            self.closed_nodes.append(node)  # for animation/plotting

            if type(self.motion_primitive_graph) is MotionPrimitiveLattice and self.goal_tolerance.size == 0:
                mp = self.motion_primitive_graph.motion_primitive_type(
                    node.state, self.goal_state, self.num_dims, self.motion_primitive_graph.max_state, self.motion_primitive_graph.mp_subclass_specific_data)
                if self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size):
                    print("Path found")
                    last_node = Node(mp.cost, 0, self.goal_state, node.state, mp)
                    self.node_dict[last_node.state.tobytes()] = last_node
                    self.build_path(last_node)
                    break

            else:
                # If node is the goal node, return path.
                if self.n == 3:  # Hack for ReedShepp
                    state = np.zeros(self.n+1)
                    state[:self.n] = node.state - self.goal_state
                    norm = np.linalg.norm(state.reshape(self.control_space_q, self.num_dims), axis=1)
                else:
                    # norm = np.linalg.norm((node.state - self.goal_state).reshape(self.control_space_q, self.num_dims), axis=1)
                    # sikang inf norm ...
                    norm = np.max(abs(node.state - self.goal_state).reshape(self.control_space_q, self.num_dims), axis=1)
                if (norm <= self.goal_tolerance[:self.control_space_q]).all():
                    print("Path found")
                    self.build_path(node)
                    break

            # JUST FOR TESTING
            if self.num_collision_checks > 100000:
                break
            # if node.graph_depth > 2:
            #     break
            # if len(self.queue) > 1000:
            #     break
            # print(node)

            neighbors = self.get_neighbor_nodes(node)
            for neighbor_node in neighbors:
                old_neighbor = self.node_dict.get(neighbor_node.state.tobytes(), None)
                if old_neighbor == None or neighbor_node.g < old_neighbor.g:
                    heappush(self.queue, neighbor_node)
                    self.node_dict[neighbor_node.state.tobytes()] = neighbor_node
                    if old_neighbor != None:
                        old_neighbor.is_closed = True
                self.neighbor_nodes.append(neighbor_node)  # for plotting

        if self.queue is not None:
            # print()
            # print(f"Nodes in queue at finish: {len(self.queue)}")
            # print(f"Closed nodes in queue at finish: {sum(node.is_closed for node in self.queue)}")
            # print()
            print(f"Nodes expanded: {self.nodes_expanded}, Path cost: {self.path_cost}")
            print(f"Number of collision checks: {self.num_collision_checks}")
            self.neighbor_nodes = np.array(self.neighbor_nodes)
            self.closed_nodes = np.array(self.closed_nodes)

        if len(self.mp_list) == 0:
            print("No path found")
        else:
            self.succeeded = True

    def expand_all_nodes(self, max_depth, plot=False):
        self.reset_graph_search()
        node = heappop(self.queue)
        neighbors = self.get_neighbor_nodes(node)
        # colors = plt.cm.tab10(np.linspace(0, 1, 10))
        vertices = [node]
        while neighbors:
            neighbor_node = neighbors.pop(0)
            if neighbor_node.graph_depth > max_depth:
                break

            new_neighbors = self.get_neighbor_nodes(neighbor_node)
            neighbors = neighbors + new_neighbors
            if plot:
                plt.plot(neighbor_node.state[0], neighbor_node.state[1], 'go',
                         zorder=10-neighbor_node.graph_depth)
                # plt.plot(neighbor_node.state[0], neighbor_node.state[1], '*',
                #         color=colors[neighbor_node.graph_depth], zorder=10-neighbor_node.graph_depth)
            if neighbor_node.mp is not None:
                if plot:
                    neighbor_node.mp.plot(position_only=True)
                vertices.append(neighbor_node)
        return vertices

    def animation_helper(self, i, closed_set_states):
        print(f"frame {i+1}/{len(self.closed_nodes)+10}")
        for k in range(len(self.lines[0])):
            self.lines[0][k].set_data([], [])

        if i >= len(self.closed_nodes):
            if type(self.motion_primitive_graph) is MotionPrimitiveLattice:
                node = self.node_dict[self.goal_state.tobytes()]
            else:
                node = self.closed_nodes[-1]
            _, sampled_path, _ = self.build_path(node)
            self.lines[0][0].set_data(sampled_path[0, :], sampled_path[1, :])
            self.lines[0][0].set_linewidth(2)
            self.lines[0][0].set_zorder(11)
        else:
            node = self.closed_nodes[i]
            open_list = []

            if type(self.motion_primitive_graph) is MotionPrimitiveLattice:
                iterator = enumerate(self.motion_primitive_graph.get_neighbor_mps(node.index))
            elif type(self.motion_primitive_graph) is MotionPrimitiveTree:
                iterator = enumerate(self.motion_primitive_graph.get_neighbor_mps(node.state, self.dt, self.num_u_per_dimension))
            for j, mp in iterator:
                mp.translate_start_position(node.state[:self.num_dims])
                if self.map.is_mp_collision_free(mp, step_size=self.mp_sampling_step_size):
                    _, sp = mp.get_sampled_position()
                    open_list.append(sp[:, -1])
                    self.lines[0][j].set_data(sp[0, :], sp[1, :])
            if open_list != []:
                self.open_list_states_animation = np.vstack((self.open_list_states_animation, np.array(open_list)))
                self.lines[4].set_data(self.open_list_states_animation[:, 0], self.open_list_states_animation[:, 1])
        self.lines[3].set_data(closed_set_states[0, :i+1, ], closed_set_states[1, :i+1])
        return self.lines

    def make_graph_search_animation(self, save_animation=False):
        plt.close('all')
        if self.queue is None:
            return
        if save_animation:
            import matplotlib
            normal_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        f, ax = plt.subplots(1, 1)
        self.map.plot(ax=ax)
        ax.axis('equal')

        mp_lines = []
        for j in range(self.num_mps):
            mp_lines.append(ax.plot([], [], linewidth=.4)[0])
        start_line, = ax.plot(self.start_state[0], self.start_state[1], 'og', zorder=4)
        goal_line, = ax.plot(self.goal_state[0], self.goal_state[1], 'or', zorder=4)
        closed_set_line, = ax.plot([], [], 'm*', zorder=3)
        open_set_line, = ax.plot([], [], '.', color=('.8'),  zorder=2)
        if self.goal_tolerance.size > 1:
            circle = plt.Circle(self.goal_state[:self.num_dims], self.goal_tolerance[0], color='b', fill=False, zorder=4)
            circle_patch = ax.add_artist(circle)
        else:
            circle_patch = None
        self.lines = [mp_lines, start_line, goal_line, closed_set_line, open_set_line, circle_patch]
        closed_set = np.array([node.state for node in self.closed_nodes]).T
        self.open_list_states_animation = self.start_state[:self.num_dims]
        ani = animation.FuncAnimation(f, self.animation_helper, len(self.closed_nodes)+10,
                                      interval=100, fargs=(closed_set,), repeat=False)

        if save_animation:
            print("Saving animation to disk")
            ani.save('graph_search.mp4')
            print("Finished saving animation")
            matplotlib.use(normal_backend)
        else:
            plt.show(block=False)
            plt.pause((len(self.closed_nodes)+10)/10)


if __name__ == "__main__":
    from motion_primitives_py import *
    import time
    # from pycallgraph import PyCallGraph, Config
    # from pycallgraph.output import GraphvizOutput
    import rospkg

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('motion_primitives')
    pkg_path = f'{pkg_path}/motion_primitives_py/motion_primitives_py/'
    mpl = MotionPrimitiveLattice.load(
        f"{pkg_path}data/lattices/testing/5.json")

    start_state = np.zeros((mpl.n))
    goal_state = np.zeros_like(start_state)

    print([len([j for j in i if j != None]) for i in mpl.edges.T])

    # occ_map = OccupancyMap.fromVoxelMapBag('data/maps/test2d.bag')
    # start_state[0:2] = [0, -18]
    # goal_state[0:2] = [7, -7]
    # goal_state[0:2] = [5, 4]
    # bag_name = f'{pkg_path}data/maps/random/trees_dispersion_1.1_54.png.bag'
    bag_name = f'{pkg_path}data/maps/trees_dispersion_0.6.bag'
    # bag_name = f'data/maps/random/trees_long0.4_13.png.bag'
    occ_map = OccupancyMap.fromVoxelMapBag(bag_name, force_2d=True)
    start_state = np.zeros((mpl.n))
    goal_state = np.zeros_like(start_state)
    start_state[0:2] = [2, 6]
    goal_state[0:2] = [48, 6]
    goal_tolerance = []  # np.ones_like(start_state)*occ_map.resolution*5

    # mpt = MotionPrimitiveTree(mpl.control_space_q, mpl.num_dims,  mpl.max_state, InputsMotionPrimitive, plot=False)
    # mpt.mp_subclass_specific_data['dt'] = .3
    # # int(np.ceil(np.sqrt(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))))
    # mpt.mp_subclass_specific_data['num_u_per_dimension'] = 4
    # mpt.mp_subclass_specific_data['rho'] = mpl.mp_subclass_specific_data['rho']

    fig, ax = plt.subplots(1, 1, sharex=True)

    # gs = GraphSearch.from_yaml("data/maps/corridor.yaml", mpt, heuristic='min_time')
    # path, sampled_path, path_cost, nodes_expanded = gs.run_graph_search()
    # gs.plot(path, sampled_path, path_cost, ax[0])

    # gs = GraphSearch.from_yaml("data/maps/corridor.yaml", mpl, heuristic='min_time')
    # mpl.mp_subclass_specific_data['iterative_bvp_dt'] = .5
    # mpl.mp_subclass_specific_data['iterative_bvp_max_t'] = 10

    gs = GraphSearch(mpl, occ_map, start_state, goal_state, heuristic='bvp', goal_tolerance=goal_tolerance)
    gs.run_graph_search()
    gs.plot(ax)

    # mpl = MotionPrimitiveLattice.load(
    #     f"{pkg_path}data/lattices/lattice_test.json")
    # gs = GraphSearch(mpl, occ_map, start_state, goal_state, heuristic='bvp', goal_tolerance=goal_tolerance)
    # gs.run_graph_search()
    # gs.plot(ax[1])

    # plt.savefig(f"plots/corridor.png", dpi=1200, bbox_inches='tight')

    plt.show()
