// Copyright 2021 Laura Jarin-Lipschitz
#include "motion_primitives/graph_search.h"

#include <ros/console.h>
#include <ros/init.h>  // ok()
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace motion_primitives {

double Elapsed(const boost::timer::cpu_timer& timer) noexcept {
  return timer.elapsed().wall / 1e9;
}

bool StatePosWithin(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
                    int spatial_dim, double d) noexcept {
  return (p1.head(spatial_dim) - p2.head(spatial_dim)).squaredNorm() < (d * d);
}

bool StateVelWithin(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2,
                    int spatial_dim, double d) noexcept {
  return (p1.segment(spatial_dim, spatial_dim) -
          p2.segment(spatial_dim, spatial_dim))
             .squaredNorm() < (d * d);
}

GraphSearch::GraphSearch(const MotionPrimitiveGraph& graph,
                         const kr_planning_msgs::VoxelMap& voxel_map,
                         const Option& options)
    : graph_(graph), voxel_map_(voxel_map), options_(options) {
  map_dims_[0] = voxel_map_.dim.x;
  map_dims_[1] = voxel_map_.dim.y;
  map_dims_[2] = voxel_map_.dim.z;
  map_origin_[0] = voxel_map_.origin.x;
  map_origin_[1] = voxel_map_.origin.y;
  map_origin_[2] = voxel_map_.origin.z;
  heuristic_types_map_["zero"] =
      &motion_primitives::GraphSearch::ComputeHeuristicZero;
  heuristic_types_map_["ruckig_bvp"] = &GraphSearch::ComputeHeuristicRuckigBVP;
  heuristic_types_map_["min_time"] = &GraphSearch::ComputeHeuristicMinTime;
  heuristic_types_map_["eth_bvp"] = &GraphSearch::ComputeHeuristicETHBVP;
  ROS_INFO("Heuristic type: %s", options_.heuristic.c_str());
  ROS_INFO("Access graph: %d", options_.access_graph);
  if (heuristic_types_map_.count(options_.heuristic) == 0) {
    ROS_ERROR("Heuristic type invalid");
  }
}

Eigen::Vector3i GraphSearch::get_indices_from_position(
    const Eigen::Vector3d& position) const {
  return floor(((position - map_origin_) / voxel_map_.resolution).array())
      .cast<int>();
}

int GraphSearch::get_linear_indices(const Eigen::Vector3i& indices) const {
  return indices[0] + map_dims_[0] * indices[1] +
         map_dims_[0] * map_dims_[1] * indices[2];
}

bool GraphSearch::is_valid_indices(const Eigen::Vector3i& indices) const {
  for (int i = 0; i < spatial_dim(); ++i) {
    if (indices[i] < 0 || (map_dims_[i] - indices[i]) <= 0) {
      return false;
    }
  }
  return true;
}

bool GraphSearch::is_free_and_valid_indices(
    const Eigen::Vector3i& indices) const {
  return ((is_valid_indices(indices) &&
           voxel_map_.data[get_linear_indices(indices)] <= 0) ||
          !is_valid_indices(indices));
  // 0 is free, -1 is unknown. TODO(laura): add back unknown_is_free option
}

bool GraphSearch::is_free_and_valid_position(Eigen::VectorXd position) const {
  // TODO(laura) enforce that position must be 2 or 3D
  if (position.rows() < 3) {
    position.conservativeResize(3);
    position(2) = options_.fixed_z;
  }
  return is_free_and_valid_indices(get_indices_from_position(position));
}

bool GraphSearch::is_mp_collision_free(
    const std::shared_ptr<MotionPrimitive> mp, double step_size) const {
  const Eigen::MatrixXd samples = mp->sample_positions(step_size);
  for (int i = 0; i < samples.cols(); ++i) {
    if (!is_free_and_valid_position(samples.col(i))) {
      return false;
    }
  }
  return true;
}

std::size_t VectorXdHash::operator()(const Eigen::VectorXd& vd) const noexcept {
  using std::size_t;

  // allow sufficiently close state to map to the same hash value
  const Eigen::VectorXi v = (vd * 100).cast<int>();

  size_t seed = 0;
  for (size_t i = 0; i < static_cast<size_t>(v.size()); ++i) {
    const auto elem = *(v.data() + i);
    seed ^= std::hash<int>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

auto GraphSearch::ExpandSingleNode(int index1, int index2, const Node& node,
                                   const State& goal_state) const
    -> std::pair<bool, Node> {
  Node next_node;
  auto failure = std::make_pair(false, next_node);

  // Check if requested motion primitive exists in the graph
  if (!graph_.HasEdge(index1, index2)) return failure;

  auto mp = graph_.get_mp_between_indices(index1, index2)->clone();
  mp->translate(node.state);

  // Check if already visited
  if (visited_states_.find(mp->end_state_) != visited_states_.cend())
    return failure;

  // Then check if its collision free
  if (!is_mp_collision_free(mp)) return failure;

  // This is a good next node
  next_node.state_index = index1;
  next_node.state = mp->end_state_;
  next_node.motion_cost = node.motion_cost + mp->cost_;
  next_node.heuristic_cost = ComputeHeuristic(mp->end_state_, goal_state);
  return std::make_pair(true, next_node);
}
auto GraphSearch::Expand(const Node& node, const State& goal_state) const
    -> std::vector<Node> {
  std::vector<Node> nodes;
  nodes.reserve(64);

  const int state_index = graph_.NormIndex(node.state_index);

  for (int i = 0; i < graph_.num_tiled_states(); ++i) {
    const auto& [success, next_node] =
        ExpandSingleNode(i, state_index, node, goal_state);
    if (success) nodes.push_back(next_node);
  }

  return nodes;
}

auto GraphSearch::ExpandPar(const Node& node, const State& goal_state) const
    -> std::vector<Node> {
  const int state_index = graph_.NormIndex(node.state_index);

  using PrivVec = tbb::enumerable_thread_specific<std::vector<Node>>;
  PrivVec priv_nodes;

  tbb::parallel_for(tbb::blocked_range<int>(0, graph_.num_tiled_states()),
                    [&, this](const tbb::blocked_range<int>& r) {
                      auto& local = priv_nodes.local();

                      for (int i = r.begin(); i < r.end(); ++i) {
                        const auto& [success, next_node] =
                            ExpandSingleNode(i, state_index, node, goal_state);
                        if (success) local.push_back(std::move(next_node));
                      }
                    });

  // combine
  std::vector<Node> nodes;
  nodes.reserve(64);
  for (const auto& each : priv_nodes) {
    nodes.insert(nodes.end(), each.begin(), each.end());
  }
  return nodes;
}

std::shared_ptr<MotionPrimitive> GraphSearch::GetPrimitiveBetween(
    const Node& start_node, const Node& end_node) const {
  const int start_index = graph_.NormIndex(start_node.state_index);
  auto mp = graph_.get_mp_between_indices(end_node.state_index, start_index);
  std::shared_ptr<MotionPrimitive> copy_mp = mp->clone();
  copy_mp->translate(start_node.state);
  return copy_mp;
}

auto GraphSearch::RecoverPath(const PathHistory& history,
                              const Node& end_node) const
    -> std::pair<std::vector<std::shared_ptr<MotionPrimitive>>, double> {
  std::vector<std::shared_ptr<MotionPrimitive>> path_mps;
  Node const* curr_node = &end_node;
  Node const* prev_node = &(history.at(curr_node->state).parent_node);
  std::vector<Node> path_nodes;
  while (true) {
    prev_node = &(history.at(curr_node->state).parent_node);
    if (prev_node->motion_cost == 0) break;
    path_nodes.push_back(*prev_node);
    path_mps.push_back(GetPrimitiveBetween(*prev_node, *curr_node));
    curr_node = prev_node;
  }
  path_nodes.push_back(*prev_node);
  if (options_.access_graph) {
    // path_mps.push_back(std::make_shared<RuckigMotionPrimitive>(
    //     graph_.spatial_dim_, prev_node->state, curr_node->state,
    //     graph_.max_state_));
    auto mp = graph_.createMotionPrimitivePtrFromGraph(prev_node->state,
                                                       curr_node->state);
    mp->compute(graph_.rho());
    mp->start_index_ = -1;
    mp->end_index_ = curr_node->state_index;
    path_mps.push_back(mp);
  } else {
    path_mps.push_back(GetPrimitiveBetween(*prev_node, *curr_node));
  }
  std::reverse(path_mps.begin(), path_mps.end());
  std::reverse(path_nodes.begin(), path_nodes.end());
  ROS_INFO_STREAM("Path cost: " << end_node.motion_cost);
  return std::make_pair(path_mps, end_node.motion_cost);
}

double GraphSearch::ComputeHeuristicMinTime(const State& v,
                                            const State& goal_state) const {
  const Eigen::VectorXd x = (v - goal_state).head(spatial_dim());
  // TODO(laura) [theoretical] needs a lot of improvement.
  return graph_.rho() * x.lpNorm<Eigen::Infinity>() / graph_.max_state()(1);
}

double GraphSearch::ComputeHeuristicRuckigBVP(const State& v,
                                              const State& goal_state) const {
  // TODO(laura) may be faster to directly call ruckig instead of creating a
  // useless MP
  auto mp =
      RuckigMotionPrimitive(spatial_dim(), v, goal_state, graph_.max_state_);
  return mp.cost_;
}
double GraphSearch::ComputeHeuristicETHBVP(const State& v,
                                           const State& goal_state) const {
  auto mp = ETHMotionPrimitive(spatial_dim(), v, goal_state, graph_.max_state_,
                               false);
  mp.compute(graph_.rho());
  return mp.cost_;
}

double GraphSearch::ComputeHeuristicZero(const State& v,
                                         const State& goal_state) const {
  return 0;
}

double GraphSearch::ComputeHeuristic(const State& v,
                                     const State& goal_state) const {
  auto func_pointer = heuristic_types_map_.at(options_.heuristic);
  return (this->*func_pointer)(v, goal_state);
}

auto GraphSearch::Search()
    -> std::pair<std::vector<std::shared_ptr<MotionPrimitive>>, double> {
  timings_.clear();
  visited_states_.clear();
  // Early exit if start and end positions are close
  if (StatePosWithin(options_.start_state, options_.goal_state,
                     graph_.spatial_dim(), options_.distance_threshold)) {
    ROS_WARN_STREAM("Start already within distance threshold of goal, exiting");
    return {};
  }
  if (!is_free_and_valid_position(options_.start_state.head(spatial_dim()))) {
    ROS_WARN_STREAM("Start is not free");
    return {};
  }
  if (!is_free_and_valid_position(options_.goal_state.head(spatial_dim()))) {
    ROS_WARN_STREAM("Goal is not free");
    return {};
  }

  // > for min heap
  auto node_cmp = [](const Node& n1, const Node& n2) {
    return n1.total_cost() > n2.total_cost();
  };
  using MinHeap =
      std::priority_queue<Node, std::vector<Node>, decltype(node_cmp)>;

  MinHeap pq{node_cmp};
  // Shortest path history, stores the parent node of a particular mp (int)
  // PathHistory history;

  auto [nodes, history] = AccessGraph(options_.start_state);
  for (auto node : nodes) {
    pq.push(node);
  }

  // timer
  boost::timer::cpu_timer timer;
  bool ros_ok = ros::ok() || !options_.using_ros;
  while (!pq.empty() && ros_ok) {
    Node curr_node = pq.top();

    // Check if we are close enough to the end
    if (StatePosWithin(curr_node.state, options_.goal_state,
                       graph_.spatial_dim(), options_.distance_threshold)) {
      if (options_.velocity_threshold >= 0 &&
          StateVelWithin(curr_node.state, options_.goal_state,
                         graph_.spatial_dim(), options_.velocity_threshold))
        ROS_WARN_STREAM("Motion primitive planning successful");
      ROS_INFO_STREAM("== pq: " << pq.size());
      ROS_INFO_STREAM("== hist: " << history.size());
      ROS_INFO_STREAM("== nodes: " << visited_states_.size());
      return RecoverPath(history, curr_node);
    }

    timer.start();
    pq.pop();
    timings_["astar_pop"] += Elapsed(timer);

    // Due to the immutability of std::priority_queue, we have no way of
    // modifying the priority of an element in the queue. Therefore, when we
    // push the next node into the queue, there might be duplicated nodes with
    // the same state but different costs. This could cause us to expand the
    // same state multiple times.
    // Although this does not affect the correctness of the implementation
    // (since the nodes are correctly sorted), it might be slower to repeatedly
    // expanding visited states. The timing suggest more than 80% of the time
    // is spent on the Expand(node) call. Thus, we will check here if this state
    // has been visited and skip if it has. This will save around 20%
    // computation.
    if (visited_states_.find(curr_node.state) != visited_states_.cend()) {
      continue;
    }
    // add current state to visited
    visited_states_.insert(curr_node.state);

    timer.start();
    const auto next_nodes = options_.parallel_expand
                                ? ExpandPar(curr_node, options_.goal_state)
                                : Expand(curr_node, options_.goal_state);
    timings_["astar_expand"] += Elapsed(timer);
    for (const auto& next_node : next_nodes) {
      // this is the best cost reaching this state (next_node) so far
      // could be inf if this state has never been visited
      const auto best_cost = history[next_node.state].best_cost;

      // compare reaching next_node from curr_node and mp to best cost
      if (next_node.motion_cost < best_cost) {
        timer.start();
        pq.push(next_node);
        timings_["astar_push"] += Elapsed(timer);
        history[next_node.state] = {curr_node, next_node.motion_cost};
      }
    }
  }
  if (pq.empty()) ROS_WARN_STREAM("Priority queue empty, exiting");
  if (!ros_ok) ROS_WARN_STREAM("Exiting because of ROS");
  return {};
}

std::vector<Eigen::VectorXd> GraphSearch::GetVisitedStates() const noexcept {
  return {visited_states_.cbegin(), visited_states_.cend()};
}

auto GraphSearch::AccessGraph(const State& start_state) const
    -> std::pair<std::vector<Node>, PathHistory> {
  PathHistory history;
  std::vector<Node> nodes;
  nodes.reserve(64);
  Node start_node;
  start_node.state_index = options_.start_index;
  start_node.state = start_state;
  start_node.motion_cost = 0.0;
  start_node.heuristic_cost =
      ComputeHeuristic(start_node.state, options_.goal_state);

  if (options_.access_graph) {
    int counter = 0;
    start_node.state_index = -1;
    for (int i = 0; i < graph_.vertices_.rows(); i += graph_.num_tiles_) {
      // TODO(laura) could parallelize
      State end_state = graph_.vertices_.row(i);
      // TODO(laura) decide if this is better than end_state(...) =
      // start_state(...)
      // end_state.head(spatial_dim()) = start_state.head(spatial_dim());
      end_state.head(spatial_dim()) += start_state.head(spatial_dim());
      auto mp =
          graph_.createMotionPrimitivePtrFromGraph(start_state, end_state);
      mp->compute(graph_.rho());
      Node next_node;
      next_node.state_index = i * graph_.num_tiles_;
      next_node.state = mp->end_state_;
      next_node.motion_cost = mp->cost_;
      next_node.heuristic_cost =
          ComputeHeuristic(mp->end_state_, options_.goal_state);
      if (next_node.motion_cost >= 0 && is_mp_collision_free(mp)) {
        counter++;
        nodes.push_back(next_node);
        history[next_node.state] = {start_node, next_node.motion_cost};
      }
      if (start_state == end_state) nodes.push_back(start_node);
    }
  } else {
    nodes.push_back(start_node);
  }
  if (nodes.size() == 0) ROS_ERROR("Access graph failure");
  return std::make_pair(nodes, history);
}

Eigen::MatrixXd GraphSearch::shift_polynomial(const Eigen::MatrixXd poly_coeffs,
                                              float shift) {
  // Pascal's triangle for computing the binomial theorem
  static Eigen::Matrix<int, 11, 11> combinatorials_ =
      (Eigen::Matrix<int, 11, 11>() << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 1, 0,
       0, 0, 0, 0, 0, 0, 1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 1, 5, 10, 10, 5, 1, 0,
       0, 0, 0, 0, 1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0, 1, 7, 21, 35, 35, 21, 7,
       1, 0, 0, 0, 1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0, 1, 9, 36, 84, 126, 126,
       84, 36, 9, 1, 0, 1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1)
          .finished();

  if (shift == 0) return poly_coeffs;
  int n_rows = poly_coeffs.rows();
  int n_cols = poly_coeffs.cols();
  int highest_order = n_cols - 1;
  Eigen::MatrixXd ret_coeffs = Eigen::MatrixXd::Constant(n_rows, n_cols, 0.0);

  for (int dim = 0; dim < n_rows; dim++) {
    for (int order = highest_order; order >= 0; order--) {
      for (int pos = 0; pos <= n_cols - 1 - order; pos++) {
        ret_coeffs(dim, n_cols - 1 - order) +=
            poly_coeffs(dim, pos) *
            combinatorials_(highest_order - pos, highest_order - pos - order) *
            std::pow(shift, highest_order - pos - order);
      }
    }
  }
  return ret_coeffs;
}

}  // namespace motion_primitives
