// Copyright 2021 Laura Jarin-Lipschitz
#include <gtest/gtest.h>

#include "motion_primitives/motion_primitive_graph.h"

using motion_primitives::PolynomialMotionPrimitive;
using motion_primitives::RuckigMotionPrimitive;
using motion_primitives::ETHMotionPrimitive;

namespace {
template <typename T>
class MotionPrimitiveTest : public ::testing::Test {
 protected:
  MotionPrimitiveTest() {
    Eigen::VectorXd start_state(6), end_state(6), max_state(6);
    start_state << 0, 0, 2, 2, 1, 1;
    end_state << 1, 1, 1, 1, 0, 0;
    max_state << 3, 3, 3, 3;
    mp_ = T(2, start_state, end_state, max_state);
    mp_.compute();
  }
  T mp_;

  Eigen::VectorXd start_state, end_state, max_state;
};

typedef ::testing::Types<PolynomialMotionPrimitive, RuckigMotionPrimitive,
                         ETHMotionPrimitive>
    MotionPrimitiveTypes;
TYPED_TEST_CASE(MotionPrimitiveTest, MotionPrimitiveTypes);

TYPED_TEST(MotionPrimitiveTest, TranslateTest) {
  // TEST_F(MotionPrimitiveTest, RuckigTranslateTest) {
  Eigen::VectorXd new_start(2);
  new_start << 4, 4;
  this->mp_.translate(new_start);

  Eigen::Vector4d new_start_state(4, 4, 2, 2), new_end_state(5, 5, 1, 1);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(this->mp_.start_state_[i], new_start_state[i]);
    EXPECT_EQ(this->mp_.end_state_[i], new_end_state[i]);
  }
}

class RuckigMotionPrimitiveTest : public ::testing::Test {
 protected:
  RuckigMotionPrimitiveTest() {
    Eigen::VectorXd start_state(6), end_state(6), max_state(6);
    start_state << 0, 0, 2, 2, 1, 1;
    end_state << 1, 1, 1, 1, 0, 0;
    max_state << 3, 3, 3, 3;
    mp_ = RuckigMotionPrimitive(2, start_state, end_state, max_state);
  }
  RuckigMotionPrimitive mp_;
};
TEST_F(RuckigMotionPrimitiveTest, RuckigJerksAndTimesTest) {
  mp_.compute();
  auto jerk_time_array = mp_.ruckig_traj_.get_jerks_and_times();
  for (int dim = 0; dim < 3; dim++) {
    double total_time = 0;
    for (auto dt : jerk_time_array[0 + dim * 2]) {
      total_time += dt;
    }
    EXPECT_EQ(total_time, mp_.ruckig_traj_.duration);
    EXPECT_EQ(total_time, mp_.traj_time_);
    for (auto jerk : jerk_time_array[1 + dim * 2]) {
      EXPECT_LE(jerk, mp_.max_state_[3]);
      EXPECT_GE(jerk, -mp_.max_state_[3]);
    }
  }
}

}  // namespace