#include <gtest/gtest.h>

#include "motion_primitives/motion_primitive_graph.h"
namespace {
using motion_primitives::MotionPrimitiveGraph;
using motion_primitives::read_motion_primitive_graph;

TEST(GraphTest, MotionPrimitiveFactoryTest) {
  const auto mp_graph = read_motion_primitive_graph("complex_test.json");
  Eigen::VectorXd start_state(mp_graph.state_dim());
  Eigen::VectorXd end_state(mp_graph.state_dim());
  end_state[0] = 5;
  auto mp = mp_graph.createMotionPrimitivePtrFromGraph(start_state, end_state);
  EXPECT_GE(mp->cost_, 0);
}
}  // namespace