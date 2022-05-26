// Copyright 2021 Laura Jarin-Lipschitz
#include <actionlib/server/simple_action_server.h>
#include <motion_primitives/graph_search.h>
#include <motion_primitives/utils.h>
#include <kr_planning_msgs/PlanTwoPointAction.h>
#include <kr_planning_msgs/SplineTrajectory.h>
#include <kr_planning_msgs/Trajectory.h>
#include <kr_planning_msgs/VoxelMap.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/MarkerArray.h>

#include <boost/circular_buffer.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <numeric>
namespace motion_primitives {

template <typename T>
std::ostream& operator<<(std::ostream& output,
                         boost::circular_buffer<T> const& values) {
  for (auto const& value : values) {
    output << value << "\n";
  }
  return output;
}
class PlanningServer {
 protected:
  ros::NodeHandle pnh_;
  actionlib::SimpleActionServer<kr_planning_msgs::PlanTwoPointAction> as_;
  ros::Publisher traj_vis_pub_;
  ros::Publisher spline_traj_pub_;
  ros::Publisher viz_traj_pub_;
  ros::Publisher sg_pub_;
  ros::Publisher visited_pub_;
  ros::Publisher time_pub_;
  ros::Publisher local_map_cleared_pub_;
  kr_planning_msgs::VoxelMapConstPtr voxel_map_ptr_ = nullptr;
  std::shared_ptr<motion_primitives::MotionPrimitiveGraph> graph_;
  std::shared_ptr<motion_primitives::MotionPrimitiveGraph> last_graph_;
  ros::Subscriber map_sub_;
  int seg_num_{-1};
  std::vector<std::shared_ptr<motion_primitives::MotionPrimitiveGraph>> graphs_;
  boost::circular_buffer<double> last_plan_times_;
  boost::circular_buffer<bool> last_plan_failures_;
  double planner_timeout_;
  int graph_index_;
  std::vector<std::string> graph_files_;
  bool parallel_expand_;

 public:
  explicit PlanningServer(const ros::NodeHandle& nh)
      : pnh_(nh), as_(nh, "plan_local_trajectory", false) {
    std::string graph_file;
    // pnh_.param("graph_file", graph_file, std::string());
    // ROS_INFO("Reading graph file %s", graph_file.c_str());
    // graph_ = read_motion_primitive_graph(graph_file);
    spline_traj_pub_ = pnh_.advertise<kr_planning_msgs::SplineTrajectory>(
        "trajectory", 1, true);
    viz_traj_pub_ =
        pnh_.advertise<kr_planning_msgs::Trajectory>("traj", 1, true);
    visited_pub_ =
        pnh_.advertise<visualization_msgs::MarkerArray>("visited", 1, true);
    map_sub_ =
        pnh_.subscribe("voxel_map", 1, &PlanningServer::voxelMapCB, this);
    sg_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("start_and_goal",
                                                              1, true);
    local_map_cleared_pub_ = pnh_.advertise<kr_planning_msgs::VoxelMap>(
        "local_voxel_map_cleared", 1, true);
    time_pub_ = pnh_.advertise<std_msgs::Float64>("planner_time", 1, true);

    pnh_.param("planner_timeout", planner_timeout_, 1.0);
    pnh_.param<bool>("parallel_expand", parallel_expand_, true);
    ROS_INFO_STREAM("Parallel expand: " << parallel_expand_);
    last_plan_times_ = boost::circular_buffer<double>(10);
    last_plan_failures_ = boost::circular_buffer<bool>(10);

    std::string graph_files_dir;
    pnh_.param("graph_files", graph_files_, std::vector<std::string>());
    pnh_.param("graph_files_dir", graph_files_dir, std::string());
    if (graph_files_.size() > 0) {
      for (auto filename : graph_files_) {
        auto full_filename = graph_files_dir + filename + ".json";
        ROS_INFO_STREAM("Reading graph file " << full_filename);
        graphs_.push_back(
            std::make_shared<motion_primitives::MotionPrimitiveGraph>(
                read_motion_primitive_graph(full_filename)));
      }
      graph_index_ = (int)(graph_files_.size() / 2);
      graph_ = graphs_[graph_index_];
      last_graph_ = graph_;
      ROS_INFO_STREAM("Using graph " << graph_files_[graph_index_]);
    }
    ROS_INFO("Finished reading graphs");
    as_.registerGoalCallback(boost::bind(&PlanningServer::executeCB, this));
    as_.start();
  }

  ~PlanningServer(void) {}

  bool is_outside_map(const Eigen::Vector3i& pn, const Eigen::Vector3i& dim) {
    return pn(0) < 0 || pn(0) >= dim(0) || pn(1) < 0 || pn(1) >= dim(1) ||
           pn(2) < 0 || pn(2) >= dim(2);
  }

  void clear_footprint(kr_planning_msgs::VoxelMap& voxel_map,
                       const Eigen::Vector3d& point) {
    // TODO(laura) copy-pasted from local_plan_server
    // Clear robot footprint
    // TODO(YUEZHAN): pass robot radius as param
    // TODO(YUEZHAN): fix val_free;
    int8_t val_free = 0;
    ROS_WARN_ONCE("Value free is set as %d", val_free);
    double robot_r = 0.5;
    int robot_r_n = std::ceil(robot_r / voxel_map.resolution);

    std::vector<Eigen::Vector3i> clear_ns;
    for (int nx = -robot_r_n; nx <= robot_r_n; nx++) {
      for (int ny = -robot_r_n; ny <= robot_r_n; ny++) {
        for (int nz = -robot_r_n; nz <= robot_r_n; nz++) {
          clear_ns.push_back(Eigen::Vector3i(nx, ny, nz));
        }
      }
    }

    auto origin_x = voxel_map.origin.x;
    auto origin_y = voxel_map.origin.y;
    auto origin_z = voxel_map.origin.z;
    Eigen::Vector3i dim = Eigen::Vector3i::Zero();
    dim(0) = voxel_map.dim.x;
    dim(1) = voxel_map.dim.y;
    dim(2) = voxel_map.dim.z;
    auto res = voxel_map.resolution;
    const Eigen::Vector3i pn =
        Eigen::Vector3i(std::round((point(0) - origin_x) / res),
                        std::round((point(1) - origin_y) / res),
                        std::round((point(2) - origin_z) / res));

    for (const auto& n : clear_ns) {
      Eigen::Vector3i pnn = pn + n;
      int idx_tmp = pnn(0) + pnn(1) * dim(0) + pnn(2) * dim(0) * dim(1);
      if (!is_outside_map(pnn, dim) && voxel_map.data[idx_tmp] != val_free) {
        voxel_map.data[idx_tmp] = val_free;
      }
    }
  }

  void adaptivePlanner() {
    if (graphs_.size() <= 1) return;
    ROS_WARN("adaptive planner");

    int num_planner_timeouts = 0;
    for (auto time : last_plan_times_) {
      if (time > planner_timeout_) num_planner_timeouts++;
    }
    bool too_many_timeouts = (num_planner_timeouts > 0);
    if (too_many_timeouts) {
      ROS_WARN("Too many planner timeouts, increasing dispersion");
      last_plan_failures_.clear();
      last_plan_times_.clear();
      graph_index_--;
    }

    int number_of_failures = std::accumulate(last_plan_failures_.begin(),
                                             last_plan_failures_.end(), 0);
    ROS_INFO_STREAM("Number of plan failures: " << number_of_failures << "/"
                                                << last_plan_failures_.size());
    bool too_many_failures = (number_of_failures > 1);
    if (too_many_failures) {
      ROS_WARN("Too many planner failures, decreasing dispersion");
      last_plan_failures_.clear();
      last_plan_times_.clear();
      graph_index_++;
    }

    if (!too_many_failures && !too_many_failures &&
        last_plan_failures_.size() > 9) {
      if (number_of_failures == 0) {
        bool well_below_planner_timeout = true;
        for (auto time : last_plan_times_) {
          if (time > planner_timeout_ * .5) {
            well_below_planner_timeout = false;
            break;
          }
        }
        if (well_below_planner_timeout) {
          ROS_WARN(
              "Planner performing well within margin, reducing dispersion.");
          last_plan_failures_.clear();
          last_plan_times_.clear();
          graph_index_++;
        }
      }
    }

    if (graph_index_ < 0) {
      graph_index_ = 0;
      ROS_WARN("Out of higher dispersion graphs");
    }
    if (graph_index_ >= graphs_.size()) {
      graph_index_ = graphs_.size() - 1;
      ROS_WARN("Out of lower dispersion graphs");
    }
    graph_ = graphs_[graph_index_];
    ROS_INFO_STREAM("Using graph " << graph_files_[graph_index_]);
  }

  typedef boost::math::tools::polynomial<double> Poly;
  Poly differentiate(const Poly& p) {
    // differentiates polynomial
    Poly::size_type rows = p.size();
    if (rows <= 1) return Poly(0.0);
    std::vector<double> v;
    for (Poly::size_type i = 1; i < rows; i++) {
      double val = static_cast<double>(i);
      v.push_back(p[i] * val);
    }
    Poly result(v.data(), rows - 2);
    return result;
  }

  std::shared_ptr<MotionPrimitive> recover_mp_from_SplineTrajectory(
      const kr_planning_msgs::SplineTrajectory& traj,
      std::shared_ptr<MotionPrimitiveGraph> graph, int seg_num) {
    Eigen::VectorXd start(graph->state_dim());
    Eigen::VectorXd end = start;
    for (int i = 0; i < graph->spatial_dim(); i++) {
      Poly const poly(traj.data[i].segs[seg_num].coeffs.begin(),
                      traj.data[i].segs[seg_num].coeffs.end());
      end[i] = poly.evaluate(1);
      auto first_deriv = differentiate(poly);
      end[i + graph->spatial_dim()] =
          first_deriv.evaluate(1) * 1. / (traj.data[i].segs[seg_num].dt);
      start[i] = traj.data[i].segs[seg_num].coeffs[0];
      start[i + graph->spatial_dim()] = traj.data[i].segs[seg_num].coeffs[1] *
                                        1. / (traj.data[i].segs[seg_num].dt);
      if (graph->control_space_dim() > 2) {
        start[i + 2 * graph->spatial_dim()] =
            traj.data[i].segs[seg_num].coeffs[2] *
            std::pow(1. / (traj.data[i].segs[seg_num].dt), 2);
        end[i + 2 * graph->spatial_dim()] =
            differentiate(first_deriv).evaluate(1) *
            std::pow(1. / (traj.data[i].segs[seg_num].dt), 2);
      }
    }
    auto mp = graph_->createMotionPrimitivePtrFromGraph(start, end);
    // mp->compute(graph_->rho());  // could copy poly_coeffs or do this

    // copy poly_coeffs
    int degree = traj.data[0].segs[seg_num].coeffs.size();
    mp->poly_coeffs_.resize(graph->spatial_dim(), degree);
    for (int i = 0; i < graph->spatial_dim(); i++) {
      for (int j = 0; j < degree; j++) {
        mp->poly_coeffs_(i, degree - j - 1) =
            traj.data[i].segs[seg_num].coeffs[j] /
            (std::pow(traj.data[0].segs[seg_num].dt, j));
      }
    }
    mp->start_index_ = traj.data[0].segs[seg_num].start_index;
    mp->end_index_ = traj.data[0].segs[seg_num].end_index;
    mp->traj_time_ = traj.data[0].segs[seg_num].dt;
    return mp;
  }
  void executeCB() {
    const kr_planning_msgs::PlanTwoPointGoal::ConstPtr& msg =
        as_.acceptNewGoal();
    kr_planning_msgs::VoxelMap voxel_map = *voxel_map_ptr_;
    if (voxel_map.resolution == 0.0) {
      ROS_ERROR(
          "Missing voxel map for motion primitive planner, aborting action "
          "server.");
      as_.setAborted();
      return;
    }
    std::array<Eigen::VectorXd, 2> start_and_goal = populateStartGoal(msg);
    Eigen::VectorXd start, goal;
    start = start_and_goal[0];
    goal = start_and_goal[1];

    ROS_INFO_STREAM("Planner start: " << start.transpose());
    ROS_INFO_STREAM("Planner goal: " << goal.transpose());

    const kr_planning_msgs::SplineTrajectory last_traj = msg->last_traj;
    double eval_time = msg->eval_time;
    double tol_pos, tol_vel;
    pnh_.param("trajectory_planner/tol_pos", tol_pos, 0.5);
    ROS_INFO_STREAM("Position tolerance: " << tol_pos);
    pnh_.param("trajectory_planner/global_goal_tol_vel", tol_vel, 1.5);
    if (msg->check_vel) ROS_INFO_STREAM("Velocity tolerance: " << tol_vel);
    std::string heuristic;
    pnh_.param<std::string>("heuristic", heuristic, "min_time");

    int planner_start_index = -1;
    bool compute_first_mp = last_traj.data.size() > 0;
    if (!compute_first_mp)
      ROS_WARN("Missing last trajectory (stopping policy may have run)");
    int seg_num = -1;
    double finished_segs_time = 0;
    std::shared_ptr<MotionPrimitive> mp;

    // TODO(laura) check first_mp for collision and if not, should access
    // graph with BVP

    // If we received a last trajectory, we want to use it to create a new
    // trajectory that starts on with the segment of the old trajectory that
    // we are flying on, at the requested eval time (which is a little bit in
    // the future of where its currently flying). With a slight abuse of
    // notation, I will refer to this as the "current segment"
    if (compute_first_mp) {
      last_graph_ = graphs_[last_traj.graph_index];

      // Figure out which segment of the last trajectory is the current
      // segment
      for (int i = 0; i < last_traj.data[0].segments; i++) {
        auto seg = last_traj.data[0].segs[i];
        finished_segs_time += seg.dt;
        if (finished_segs_time > eval_time) {
          finished_segs_time -= seg.dt;
          seg_num = i;
          break;
        }
      }
      // todo(laura) what to do if eval time is past the end of the traj
      if (seg_num == -1) seg_num = last_traj.data[0].segments - 1;
      ROS_INFO_STREAM("Seg num " << seg_num);

      // The planner will start at the end of the current segment, so we get
      // its end index.
      planner_start_index = last_traj.data[0].segs[seg_num].end_index;

      mp = recover_mp_from_SplineTrajectory(last_traj, last_graph_, seg_num);

      start_and_goal[0] = mp->end_state_;
      ROS_INFO_STREAM("Planner adjusted start: " << mp->end_state_.transpose());
    }

    GraphSearch::Option options = {.start_state = start_and_goal[0],
                                   .goal_state = goal,
                                   .distance_threshold = tol_pos,
                                   .parallel_expand = parallel_expand_,
                                   .heuristic = heuristic,
                                   .access_graph = false,
                                   .start_index = planner_start_index};
    if (graph_->spatial_dim() == 2) options.fixed_z = msg->p_init.position.z;
    if (msg->check_vel) options.velocity_threshold = tol_vel;
    if (planner_start_index == -1 ||
        planner_start_index >= graph_->num_tiled_states())
      options.access_graph = true;
    // TODO(laura) why should planner_start_index >= graph_->num_tiled_states()
    // ever happen except when switching graphs, but checking if
    // graph_index_!=last_graph.graph_index didn't work

    publishStartAndGoal(start_and_goal, options.fixed_z);
    Eigen::Vector3d map_start;
    map_start(0) = start(0);
    map_start(1) = start(1);
    map_start(2) = msg->p_init.position.z;
    // Sets the voxel_map_ start to be collision free
    clear_footprint(voxel_map, map_start);

    // If we are not at the end, we can also clear the local goal
    if (!msg->check_vel) {
      Eigen::Vector3d map_goal;
      map_goal(0) = goal(0);
      map_goal(1) = goal(1);
      map_goal(2) = msg->p_init.position.z;
      clear_footprint(voxel_map, map_goal);
      // local_map_cleared_pub_.publish(voxel_map);
    }

    GraphSearch gs(*graph_, voxel_map, options);
    const auto start_time = ros::Time::now();

    auto [path, nodes] = gs.Search();
    int last_graph_index = graph_index_;
    adaptivePlanner();
    bool planner_start_too_close_to_goal =
        StatePosWithin(options.start_state, options.goal_state,
                       graph_->spatial_dim(), options.distance_threshold);
    if (path.empty()) {
      if (!planner_start_too_close_to_goal) {
        ROS_ERROR("Graph search failed, aborting action server.");
        last_plan_failures_.push_back(true);
        as_.setAborted();
        return;
      }
    }
    if (compute_first_mp) {
      // To our planned trajectory, we add a cropped motion primitive that is
      // in the middle of the last_traj segment that we are evaluating at

      Eigen::VectorXd cropped_start(graph_->state_dim());
      Eigen::MatrixXd cropped_poly_coeffs;

      double shift_time = 0;
      // We need to shift the motion primitive to start at the eval_time. If
      // we are the first segment, it might have been itself a cropped motion
      // primitive, so the shift_time is a little more complicated.
      // finished_segs_time is the total time of all the segments before the
      // current segment.
      if (seg_num == 0)
        shift_time = eval_time;
      else
        shift_time = eval_time - finished_segs_time;

      cropped_poly_coeffs =
          GraphSearch::shift_polynomial(mp->poly_coeffs_, shift_time);

      // Use the polynomial coefficients to get the new start, which is
      // hopefully the same as the planning query requested start.
      for (int i = 0; i < graph_->spatial_dim(); i++) {
        cropped_start(i) =
            cropped_poly_coeffs(i, cropped_poly_coeffs.cols() - 1);
        cropped_start(graph_->spatial_dim() + i) =
            cropped_poly_coeffs(i, cropped_poly_coeffs.cols() - 2);
        if (graph_->control_space_dim() > 2) {
          cropped_start(2 * graph_->spatial_dim() + i) =
              cropped_poly_coeffs(i, cropped_poly_coeffs.cols() - 3);
        }
      }
      ROS_INFO_STREAM("Cropped start " << cropped_start.transpose());

      auto first_mp =
          graph_->createMotionPrimitivePtrFromGraph(cropped_start, start);
      // auto first_mp = graph_->createMotionPrimitivePtrFromGraph(
      //     cropped_start, path[0]->start_state_);

      double new_seg_time = mp->traj_time_ - shift_time;
      first_mp->populate(0, new_seg_time, cropped_poly_coeffs,
                         last_traj.data[0].segs[seg_num].start_index,
                         last_traj.data[0].segs[seg_num].end_index);
      // Add the cropped motion primitive to the beginning of the planned
      // trajectory
      path.insert(path.begin(), first_mp);
    }

    ROS_INFO("Graph search succeeded.");
    last_plan_failures_.push_back(false);

    const auto total_time = (ros::Time::now() - start_time).toSec();
    ROS_INFO("Finished planning. Planning time %f s", total_time);
    last_plan_times_.push_back(total_time);
    std_msgs::Float64 time;
    time.data = total_time;
    time_pub_.publish(time);

    kr_planning_msgs::PlanTwoPointResult result;
    result.epoch = msg->epoch;
    result.execution_time = msg->execution_time;
    result.traj = kr_planning_msgs::SplineTrajectory();
    result.traj.header.stamp = ros::Time::now();
    result.traj.header.frame_id = voxel_map.header.frame_id;
    result.success = true;
    result.traj = path_to_spline_traj_msg(path, result.traj.header,
                                          msg->p_init.position.z);
    result.traj.graph_index = last_graph_index;
    spline_traj_pub_.publish(result.traj);
    if (path[0]->poly_coeffs_.size() <= 6) {
      auto viz_traj_msg =
          path_to_traj_msg(path, result.traj.header, msg->p_init.position.z);
      viz_traj_pub_.publish(viz_traj_msg);
    }
    as_.setSucceeded(result);

    ROS_INFO_STREAM("path size: " << path.size());

    const auto visited_marray = StatesToMarkerArray(
        gs.GetVisitedStates(), gs.spatial_dim(), voxel_map.header);
    visited_pub_.publish(visited_marray);
  }

  void voxelMapCB(const kr_planning_msgs::VoxelMap::ConstPtr& msg) {
    voxel_map_ptr_ = msg;
  }

  std::array<double, 3> pointMsgToArray(const geometry_msgs::Point& point) {
    return {point.x, point.y, point.z};
  }

  std::array<double, 3> pointMsgToArray(const geometry_msgs::Vector3& point) {
    return {point.x, point.y, point.z};
  }
  std::array<Eigen::VectorXd, 2> populateStartGoal(
      const kr_planning_msgs::PlanTwoPointGoal::ConstPtr& msg) {
    Eigen::VectorXd start, goal;
    start.resize(graph_->state_dim());
    goal.resize(graph_->state_dim());
    auto p_init = pointMsgToArray(msg->p_init.position);
    auto v_init = pointMsgToArray(msg->v_init.linear);
    auto a_init = pointMsgToArray(msg->a_init.linear);
    auto p_final = pointMsgToArray(msg->p_final.position);
    auto v_final = pointMsgToArray(msg->v_final.linear);
    auto a_final = pointMsgToArray(msg->a_final.linear);

    int spatial_dim = graph_->spatial_dim();
    for (int dim = 0; dim < spatial_dim; dim++) {
      start[dim] = p_init[dim];
      goal[dim] = p_final[dim];
      start[spatial_dim + dim] = v_init[dim];
      goal[spatial_dim + dim] = v_final[dim];
      if (graph_->control_space_dim() > 2) {
        start[2 * spatial_dim + dim] = a_init[dim];
        goal[2 * spatial_dim + dim] = a_final[dim];
      }
    }

    std::array<Eigen::VectorXd, 2> start_and_goal{start, goal};
    return start_and_goal;
  }

  void publishStartAndGoal(const std::array<Eigen::VectorXd, 2>& start_and_goal,
                           double fixed_z) {
    visualization_msgs::MarkerArray sg_markers;
    visualization_msgs::Marker start_marker, goal_marker;
    start_marker.header = voxel_map_ptr_->header;
    start_marker.pose.position.x = start_and_goal[0][0],
    start_marker.pose.position.y = start_and_goal[0][1];
    start_marker.pose.orientation.w = 1;
    start_marker.color.g = 1;
    start_marker.color.a = 1;
    start_marker.type = 2;
    start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 1;
    goal_marker = start_marker;
    goal_marker.id = 1;
    goal_marker.pose.position.x = start_and_goal[1][0],
    goal_marker.pose.position.y = start_and_goal[1][1];

    if (graph_->spatial_dim() == 2) {
      start_marker.pose.position.z = fixed_z;
      goal_marker.pose.position.z = fixed_z;
    } else {
      start_marker.pose.position.z = start_and_goal[0][2];
      goal_marker.pose.position.z = start_and_goal[1][2];
    }
    goal_marker.color.b = 1;
    start_marker.color.a = 1;
    sg_markers.markers.push_back(start_marker);
    sg_markers.markers.push_back(goal_marker);
    sg_pub_.publish(sg_markers);
  }
};

}  // namespace motion_primitives

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "motion_primitives_action_server");
  ros::NodeHandle pnh("~");
  motion_primitives::PlanningServer ps(pnh);
  ros::spin();
}
