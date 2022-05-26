from motion_primitives_py import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import rospkg
import os
import motion_primitives_cpp
from pylab import rcParams

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('motion_primitives')
pkg_path = f'{pkg_path}/motion_primitives_py/motion_primitives_py/'

dispersion_threshholds = [300, 400, 500, 600, 700]  # list(np.arange(1, 201))
file_prefix = f'{pkg_path}data/lattices/eth2/eth'
params = {
    'font.size': 12,
    'legend.fontsize': 9,
    'text.usetex': False,
    'figure.figsize': [7, 6]
}

rcParams.update(params)


def generate_data():
    # get all lattices in directory
    # file_prefix = f'{pkg_path}data/lattices/dispersion'
    mpls = {}
    for dispersion_threshhold in deepcopy(dispersion_threshholds):
        filename = f"{file_prefix}{dispersion_threshhold}"
        print(f"{filename}.json")
        try:
            mpls[dispersion_threshhold] = motion_primitives_cpp.read_motion_primitive_graph(f"{filename}.json")
            # mpls[dispersion_threshhold] = MotionPrimitiveLattice.load(f"{filename}.json")
        except:
            print("No lattice file")
            dispersion_threshholds.remove(dispersion_threshhold)

    data_array = np.zeros((3, 100, len(dispersion_threshholds)))
    for n in range(1, 99):  # iterate over maps
        # bag_name = f'{pkg_path}data/maps/png_maps/trees_long4.5_6.png.bag'
        bag_name = f'{pkg_path}data/maps/png_maps/trees_long0.3_{n}.png.bag'
        # bag_name = f'{pkg_path}data/maps/clutteredness/trees_long3_{n}.png.bag'
        # bag_name = f'data/maps/random/trees_dispersion_0.6_{n}.png.bag'
        # occ_map = OccupancyMap.fromVoxelMapBag(bag_name, force_2d=True)
        start_state = np.zeros((6))
        goal_state = np.zeros_like(start_state)
        start_state[0:2] = [5, 10]
        goal_state[0:2] = [55, 50]

        voxel_map = motion_primitives_cpp.read_bag(bag_name, "/voxel_map", 0)[-1]

        for i, dispersion_threshhold in enumerate(dispersion_threshholds):  # iterate over lattices
            mpl = mpls[dispersion_threshhold]
            print(dispersion_threshhold)
            # gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n],
            #                  heuristic='min_time', mp_sampling_step_size=occ_map.resolution/mpl.max_state[1], goal_tolerance=np.ones(mpl.n))
            # gs.run_graph_search()
            option = motion_primitives_cpp.Option()
            option.start_state = start_state
            option.goal_state = goal_state
            option.parallel_expand = True
            option.access_graph = False
            option.using_ros = False
            option.distance_threshold = 5
            gs = motion_primitives_cpp.GraphSearch(mpl, voxel_map, option)
            x = gs.Search()
            data_array[0, n-1, i] = x
            data_array[1, n-1, i] = 0  # gs.nodes_expanded
            data_array[2, n-1, i] = gs.num_visited()
            # gs.plot()
            # plt.show()
    np.save('random_data', data_array)


def process_data():
    data = np.load('random_data.npy')
    for dispersion_threshhold in deepcopy(dispersion_threshholds):
        filename = f"{file_prefix}{dispersion_threshhold}"
        print(f"{filename}.json")
        try:
            open(f"{filename}.json")
        except:
            print("No lattice file")
            dispersion_threshholds.remove(dispersion_threshhold)
    fig, ax = plt.subplots(2, 1, sharex=True)
    path_cost = data[0, :, :]
    average_path_cost = np.nanmean(path_cost, axis=0)
    dispersions = [213.037, 185.24585, 168.5891, 154.450681, 142.816]

    ax[0].plot(dispersions, average_path_cost, 'o')
    ax[0].set_ylabel("Cost")
    error = np.nanstd(path_cost, axis=0)
    ax[0].fill_between(dispersions, average_path_cost-error, average_path_cost+error,
                       alpha=0.5, label="1 Std. Deviation")
    ax[0].legend()
    nodes_expanded = data[1, :, :]
    # average_nodes_expanded = np.average(nodes_expanded, axis=0)
    # ax[2].plot(dispersion_threshholds, average_nodes_expanded)
    # # ax[1].xlabel("Dispersion")
    # ax[2].set_ylabel("Nodes Expanded")

    nodes_considered = data[2, :, :]
    # nodes_considered[nodes_expanded > 999] = np.nan
    average_nodes_considered = np.nanmean(nodes_considered, axis=0)
    ax[1].plot(dispersions, average_nodes_considered, 'o')
    # ax[2].xlabel("Dispersion")

    error = np.nanstd(nodes_considered, axis=0)
    ax[1].fill_between(dispersions, average_nodes_considered-error, average_nodes_considered+error,
                       alpha=0.5)
    ax[1].set_ylabel("# of collision checks")

    ax[1].set_xlabel("Dispersion")

    plt.show()


def generate_data_clutteredness():
    # get all lattices in directory
    file_prefix = f'{pkg_path}data/lattices/dispersion'
    dispersion_threshhold = 70
    mpl = MotionPrimitiveLattice.load(f"{file_prefix}{dispersion_threshhold}.json")

    data_dict = {}
    counter = 0
    for root, dirs, files in os.walk(f'{pkg_path}data/maps/clutteredness/'):
        for f in files:
            if "bag" in f:
                counter += 1
                poisson_spacing = float(f.split("_")[1][4:])
                print(f"Poisson Spacing: {poisson_spacing}")
                voxel_map = motion_primitives_cpp.read_bag(root+f, "/voxel_map", 0)[-1]

                # occ_map = OccupancyMap.fromVoxelMapBag(root+f, force_2d=True)
                start_state = np.zeros((4))
                goal_state = np.zeros_like(start_state)
                start_state[0:2] = [2, 6]
                goal_state[0:2] = [48, 6]
                # occ_map.plot()
                # plt.show()

                gs = GraphSearch(mpl, occ_map, start_state[:mpl.n], goal_state[:mpl.n],
                                 heuristic='min_time', mp_sampling_step_size=occ_map.resolution/mpl.max_state[1], goal_tolerance=[1, 1])
                gs.run_graph_search()
                if gs.path_cost is None:
                    continue
                data_dict[poisson_spacing] = data_dict.get(poisson_spacing, None)
                d = np.array([gs.path_cost, gs.num_collision_checks]).reshape(1, 2)
                if data_dict[poisson_spacing] is None:
                    data_dict[poisson_spacing] = d
                else:
                    data_dict[poisson_spacing] = np.vstack((data_dict[poisson_spacing], d))
                if counter > 1000:
                    break

    data_processed = np.empty((len(data_dict), 5))
    for i, (k, v) in enumerate(data_dict.items()):
        data_processed[i, 0] = k
        data_processed[i, 1] = np.average(v[:, 0])
        data_processed[i, 2] = np.std(v[:, 0])
        data_processed[i, 3] = np.average(v[:, 1])
        data_processed[i, 4] = np.std(v[:, 1])
    np.save("cluttered_data", data_processed)
    plt.show()


def process_data_clutteredness():
    data_processed = np.load("cluttered_data.npy")
    fig, ax = plt.subplots(2, 1, sharex=True)
    print(data_processed[:, 0])
    print(data_processed[:, 1])
    ax[0].plot(data_processed[:, 0], data_processed[:, 1], 'o')
    # ax[0].fill_between(data_processed[:,0], data_processed[:,1]-data_processed[:,2], data_processed[:,1]+data_processed[:,2],
    #                    alpha=0.5)
    ax[1].plot(data_processed[:, 0], data_processed[:, 3], 'o')
    # ax[1].fill_between(data_processed[:,0], data_processed[:,3]-data_processed[:,4], data_processed[:,3]+data_processed[:,4],
    #                    alpha=0.5)
    ax[0].set_ylabel("Cost")
    ax[1].set_ylabel("Nodes considered")
    ax[1].set_xlabel("Poisson Spacing")
    plt.show()


if __name__ == '__main__':
    generate_data()
    process_data()
    # generate_data_clutteredness()
    # process_data_clutteredness()
    # data = np.load('data/random_data_2.npy')
    # process_data(data)
