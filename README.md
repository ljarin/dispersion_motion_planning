
## C++/ROS Installation, also some python bindings
You must have ROS and catkin tools already installed and sourced. The ROS package is independent from/does not also install standalone python package (see below). Currently, graph generation is only included in python standalone version, but graph search exists in both (but focus is on C++ version). There are limited python bindings (mostly t call GraphSearch) for the C++ version that ROS will install for the default python. This means they will only work for Python2 on 18.04 and Python3 for 20.04.

```
sudo apt-get install -y libeigen3-dev libtbb-dev libgtest-dev python3-vcstool
mkdir -p dispersion_ws/src
cd dispersion_ws
catkin init
cd src
git clone git@github.com:ljarin/motion_primitives.git
vcs import < motion_primitives/deps_ssh.repos # (or deps_https.ssh)
catkin b
```

## Python standalone installation
Install motion_primitives_py package:
```
cd motion_primitives_py
pip3 install -e .
```

If you don't have ROS and above workspace installed and sourced:
- `pip3 install --extra-index-url https://rospypi.github.io/simple/ rosbag`
- ETHMotionPrimitive will not work, but you should still be able to run everything else. To install ETHMotionPrimitive see https://github.com/ljarin/mav_trajectory_generation

System packages (for animation video mp4s to be generated):
- `sudo apt-get install ffmpeg`
