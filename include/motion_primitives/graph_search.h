// Copyright 2021 Laura Jarin-Lipschitz
#pragma once

#include <kr_planning_msgs/VoxelMap.h>

#include <boost/timer/timer.hpp>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "motion_primitives/motion_primitive_graph.h"

namespace motion_primitives {

// First convert VectorXd to VectorXi by some scaling then hash
// This is to avoid potential floating point error causing the same state to
// hash to different values
// NOTE: Ideally this should be part of the implementation, we put in the public
// namespace so that we can test it
struct VectorXdHash : std::unary_function<Eigen::VectorXd, std::size_t> {
  std::size_t operator()(const Eigen::VectorXd& vd) const noexcept;
};

double Elapsed(const boost::timer::cpu_timer& timer) noexcept;
bool StatePosWithin(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
                    int spatial_dim, double d) noexcept;

class GraphSearch {
 protected:
  MotionPrimitiveGraph graph_;
  Eigen::Vector3i map_dims_;
  Eigen::Vector3d map_origin_;
  kr_planning_msgs::VoxelMap voxel_map_;

 public:
  using State = Eigen::VectorXd;
  struct Option {
    State start_state;
    State goal_state;
    double distance_threshold;
    bool parallel_expand{false};
    std::string heuristic{"min_time"};
    bool access_graph{false};
    int start_index{0};
    double fixed_z{0};
    double velocity_threshold = -1;
    bool using_ros{true};
  };
  struct Node {
    static constexpr auto kInfCost = std::numeric_limits<double>::infinity();

    int state_index{0};  // used to retrieve mp from graph
    State state;
    double motion_cost{kInfCost};
    double heuristic_cost{0.0};

    double total_cost() const noexcept { return motion_cost + heuristic_cost; }
  };

  GraphSearch(const MotionPrimitiveGraph& graph,
              const kr_planning_msgs::VoxelMap& voxel_map,
              const Option& options);

  ~GraphSearch() = default;
  const Option options_;

  // Search for a path from start_state to end_state, stops if no path found
  // (returns empty vector) or reach within distance_threshold of start_state
  // parallel == true will expand nodes in parallel (~x2 speedup)
  std::pair<std::vector<std::shared_ptr<MotionPrimitive>>, double> Search();

  std::vector<Eigen::VectorXd> GetVisitedStates() const noexcept;
  const auto& timings() const noexcept { return timings_; }
  int spatial_dim() const noexcept { return graph_.spatial_dim_; }
  static Eigen::MatrixXd shift_polynomial(const Eigen::MatrixXd poly_coeffs,
                                          float shift);

  // State is the real node
  // Node is a wrapper around state that also carries the cost info

 private:
  // The state is the key of PathHistory and will not be stored here
  struct StateInfo {
    Node parent_node;                  // parent node of this state
    double best_cost{Node::kInfCost};  // best cost reaching this state so far
  };

  // Path history stores the parent node of this state and the best cost so far
  using PathHistory = std::unordered_map<State, StateInfo, VectorXdHash>;
  std::pair<std::vector<std::shared_ptr<MotionPrimitive>>, double> RecoverPath(
      const PathHistory& history, const Node& end_node) const;

  typedef double (motion_primitives::GraphSearch::*FUNCPTR)(
      const State& v, const State& goal_state) const;
  std::unordered_map<std::string, FUNCPTR> heuristic_types_map_;

  double ComputeHeuristic(const State& state, const State& goal_state) const;
  double ComputeHeuristicZero(const State& v, const State& goal_state) const;
  double ComputeHeuristicRuckigBVP(const State& v,
                                   const State& goal_state) const;
  double ComputeHeuristicMinTime(const State& v, const State& goal_state) const;
  double ComputeHeuristicETHBVP(const State& v, const State& goal_state) const;

  // Stores all visited states
  std::vector<Node> Expand(const Node& node, const State& goal_state) const;
  std::vector<Node> ExpandPar(const Node& node, const State& goal_state) const;
  std::pair<bool, Node> ExpandSingleNode(int index1, int index2,
                                         const Node& node,
                                         const State& goal_state) const;

  std::pair<std::vector<Node>, PathHistory> AccessGraph(
      const State& start_state) const;

  std::shared_ptr<MotionPrimitive> GetPrimitiveBetween(
      const Node& start_node, const Node& end_node) const;

  using StateSet = std::unordered_set<State, VectorXdHash>;
  StateSet visited_states_;
  // internal use only, stores (wall) time spent on different parts
  std::unordered_map<std::string, double> timings_;

  Eigen::Vector3i get_indices_from_position(
      const Eigen::Vector3d& position) const;
  // Converts from vector of indices to single index into
  // kr_planning_msgs::VoxelMap.data
  int get_linear_indices(const Eigen::Vector3i& indices) const;
  bool is_valid_indices(const Eigen::Vector3i& indices) const;
  bool is_free_and_valid_indices(const Eigen::Vector3i& indices) const;
  bool is_free_and_valid_position(Eigen::VectorXd v) const;
  // Samples motion primitive along step_size time steps and checks for
  // collisions
  bool is_mp_collision_free(const std::shared_ptr<MotionPrimitive> mp,
                            double step_size = 0.2) const;
};

}  // namespace motion_primitives
