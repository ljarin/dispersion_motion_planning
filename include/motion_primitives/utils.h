// Copyright 2021 Laura Jarin-Lipschitz
#pragma once

#include <kr_planning_msgs/SplineTrajectory.h>
#include <kr_planning_msgs/Trajectory.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/console.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <memory>
#include <vector>

#include "motion_primitives/motion_primitive_graph.h"

namespace motion_primitives {

kr_planning_msgs::Trajectory path_to_traj_msg(
    const std::vector<std::shared_ptr<MotionPrimitive>>& mps,
    const std_msgs::Header& header, float z_height = 0.0);

kr_planning_msgs::SplineTrajectory path_to_spline_traj_msg(
    const std::vector<std::shared_ptr<MotionPrimitive>>& mps,
    const std_msgs::Header& header, float z_height = 0.0);

auto StatesToMarkerArray(const std::vector<Eigen::VectorXd>& states,
                         int spatial_dim, const std_msgs::Header& header,
                         double scale = 0.1, bool show_vel = false)
    -> visualization_msgs::MarkerArray;

// num indicates the max number of elements to read, -1 means read till the end
template <class T>
std::vector<T> read_bag(std::string file_name, std::string topic,
                        unsigned int num) {
  rosbag::Bag bag;
  bag.open(file_name, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(topic);
  rosbag::View view(bag, rosbag::TopicQuery(topics));

  std::vector<T> msgs;
  BOOST_FOREACH (rosbag::MessageInstance const m, view) {
    if (m.instantiate<T>() != NULL) {
      msgs.push_back(*m.instantiate<T>());
      if (msgs.size() > num) break;
    }
  }
  bag.close();
  if (msgs.empty())
    ROS_WARN("Fail to find '%s' in '%s', make sure md5sum are equivalent.",
             topic.c_str(), file_name.c_str());
  else
    ROS_INFO("Get voxel map data!");
  return msgs;
}

}  // namespace motion_primitives
