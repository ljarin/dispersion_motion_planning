import numpy as np
import matplotlib.pyplot as plt

# from motion_primitives_py.msg import VoxelMap


class OccupancyMap():
    def __init__(self, resolution, origin, dims, data, force_2d=False, unknown_is_free=False):
        self.resolution = resolution
        self.voxels = np.squeeze(np.array(data).reshape(dims, order='F'))
        self.dims = np.array(self.voxels.shape)
        if force_2d and self.dims.shape[0]==3:
            self.dims = self.dims[:2]
            self.voxels = self.voxels[:, :, 0]
        self.origin = origin[:len(self.dims)]
        map_size = self.dims*self.resolution
        map_min = self.origin
        map_max = map_size + self.origin
        self.extent = [map_min[0], map_max[0], map_min[1], map_max[1]]
        self.unknown_is_free = unknown_is_free

    @classmethod
    def fromVoxelMapBag(cls, filename, topic=None, force_2d=False, unknown_is_free=False):
        try:
            import rosbag
        except:
            print("Error: you need to install rosbag to use this function. If ROS is sourced you do not need to do anything else. Otherwise, you can run `pip3 install --extra-index-url https://rospypi.github.io/simple/ rosbag`. Exiting.")
            raise SystemExit

        # load messages from bagfile
        bag = rosbag.Bag(filename)
        msgs = [msg for _, msg, _ in bag.read_messages(topics=topic)]
        bag.close()
        resolution = msgs[0].resolution
        dims = np.array([msgs[0].dim.x, msgs[0].dim.y, msgs[0].dim.z]).astype(int)
        origin = np.array([msgs[0].origin.x, msgs[0].origin.y, msgs[0].origin.z])
        return cls(resolution, origin, dims, np.array(msgs[0].data), force_2d=force_2d, unknown_is_free=unknown_is_free)

    def get_indices_from_position(self, point):
        return np.floor((point - self.origin) / self.resolution).astype(int)

    def get_voxel_center_from_indices(self, indices):
        return self.resolution * (indices + .5) + self.origin

    def is_valid_indices(self, indices):
        for i in range(len(self.dims)):
            if indices[i] < 0 or (self.dims[i] - indices[i]) <= 0:
                return False
        else:
            return True

    def is_valid_position(self, position):
        return self.is_valid_indices(self.get_indices_from_position(position))

    def is_free_and_valid_indices(self, indices):
        if (self.is_valid_indices(indices) and self.voxels[tuple(indices)] <= 0) or (not self.is_valid_indices(indices) and self.unknown_is_free):
            return True
        else:
            return False

    def is_free_and_valid_position(self, position):
        indices = self.get_indices_from_position(position)
        return self.is_free_and_valid_indices(indices)

    def is_mp_collision_free(self, mp, step_size=0.1):
        """
        Function to check if there is a collision between a motion primitive
        trajectory and the occupancy map

        Input:
            mp, a MotionPrimitive object to be checked
        Output:
            collision, boolean that is True if there were no collisions
        """
        if not mp.is_valid:
            return False
        _, samples = mp.get_sampled_position(step_size)
        for sample in samples.T:
            if not self.is_free_and_valid_position(sample):
                return False
        return True

    def plot(self, ax=None):
        if ax == None:
            fig, self.ax = plt.subplots()
            ax = self.ax

        ax.add_patch(plt.Rectangle(self.origin, self.dims[0]*self.resolution, self.dims[1]*self.resolution, ec='r', fill=False))
        ax.set_aspect('equal')

        if len(self.dims) == 2:
            ax.pcolormesh(np.arange(self.voxels.shape[0]+1)*self.resolution + self.origin[0], np.arange(self.voxels.shape[1]+1)
                          * self.resolution + self.origin[1], self.voxels.T, cmap='Greys', zorder=1)
        else:
            print("WARNING: cannot plot in 3D, plotting a slice in the middle")
            print(self.voxels.T.shape)
            ax.pcolormesh(np.arange(self.voxels.shape[0]+1)*self.resolution + self.origin[0], np.arange(self.voxels.shape[1]+1)
                * self.resolution + self.origin[1], self.voxels.T[int(self.voxels.shape[2]/2),:,:], cmap='Greys', zorder=1)

        return ax


if __name__ == "__main__":
    from motion_primitives_py import *
    import rospkg

    # problem parameters
    num_dims = 2
    control_space_q = 3

    # setup occupancy map
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('motion_primitives')
    occ_map = OccupancyMap.fromVoxelMapBag(f'{pkg_path}/motion_primitives_py/data/maps/test2d.bag')
    occ_map.plot()
    print(occ_map.extent)

    # setup sample motion primitive
    start_state = np.zeros((num_dims * control_space_q,))
    start_state[:2] = [18, 5]
    end_state = np.zeros((num_dims * control_space_q,))
    end_state[:2] = [10, 8]
    max_state = 1000 * np.ones((num_dims * control_space_q,))
    mp = PolynomialMotionPrimitive(start_state, end_state, num_dims, max_state)
    mp.plot(position_only=True, ax=occ_map.ax)

    print(occ_map.is_free_and_valid_position([70, 5]))
    plt.show()
