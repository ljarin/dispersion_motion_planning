
// Copyright 2021 Laura Jarin-Lipschitz
#include <gtest/gtest.h>
#include <ros/console.h>

#include "motion_primitives/graph_search.h"
using motion_primitives::GraphSearch;
using motion_primitives::read_motion_primitive_graph;
namespace {
class GraphSearchTest : public ::testing::Test {
 protected:
  GraphSearchTest() {
    voxel_map_.resolution = 1.;
    voxel_map_.dim.x = 20;
    voxel_map_.dim.y = 20;
    voxel_map_.data.resize(voxel_map_.dim.x * voxel_map_.dim.y, 0);
    Eigen::Vector2d start(3, 3);
    Eigen::Vector2d goal(5, 5);
    option_ = GraphSearch::Option{
        .start_state = start,
        .goal_state = goal,
        .distance_threshold = 0.001,
        .parallel_expand = true,
        .heuristic = "min_time",
        .access_graph = false,
        .start_index = 0,
        .fixed_z = 0,
    };
    option_.using_ros = false;
  }
  kr_planning_msgs::VoxelMap voxel_map_;
  GraphSearch::Option option_;
};

TEST_F(GraphSearchTest, OptimalPath) {
  const auto mp_graph = read_motion_primitive_graph("simple_test.json");
  GraphSearch gs(mp_graph, voxel_map_, option_);
  const auto path = gs.Search().first;

  float path_cost = 0;
  for (auto seg : path) {
    path_cost += seg->cost_;
    ROS_INFO_STREAM(seg->start_state_);
    ROS_INFO_STREAM(seg->end_state_);
  }
  EXPECT_EQ(path_cost, 2);
}

TEST_F(GraphSearchTest, ShiftPolynomial) {
  const auto mp_graph = read_motion_primitive_graph("complex_test.json");
  GraphSearch gs(mp_graph, voxel_map_, option_);
  Eigen::MatrixXd poly_coeffs(3, 5);
  poly_coeffs << 1, 2, 3, 2, 2, 2, 3, 4, 5, 6, 3, 2, 1, 1, 4;
  float t1 = 3.5;
  float t2 = 2.5;

  Eigen::VectorXd ts(5);
  for (int i = 0; i < 5; i++) {
    ts[i] = std::pow(t1, 4 - i);
  }
  Eigen::VectorXd unshifted_state = poly_coeffs * ts;

  for (int i = 0; i < 5; i++) {
    ts[i] = std::pow(t1 - t2, 4 - i);
  }

  Eigen::MatrixXd shifted_poly_coeffs = gs.shift_polynomial(poly_coeffs, t2);
  Eigen::VectorXd shifted_state = shifted_poly_coeffs * ts;
  EXPECT_EQ(unshifted_state, shifted_state);
}

}  // namespace