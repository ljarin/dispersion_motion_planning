// Copyright 2021 Laura Jarin-Lipschitz

#include "motion_primitives/motion_primitive_graph.h"

#include <mav_trajectory_generation/polynomial_optimization_nonlinear.h>
#include <kr_planning_msgs/Polynomial.h>
#include <ros/console.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <ostream>

using kr_planning_msgs::Polynomial;
using kr_planning_msgs::Spline;
namespace motion_primitives {

void MotionPrimitive::translate(const Eigen::VectorXd& new_start) {
  end_state_.head(spatial_dim_) = end_state_.head(spatial_dim_) -
                                  start_state_.head(spatial_dim_) +
                                  new_start.head(spatial_dim_);
  start_state_.head(spatial_dim_) = new_start.head(spatial_dim_);
  poly_coeffs_.col(poly_coeffs_.cols() - 1) = new_start.head(spatial_dim_);
}

void MotionPrimitive::translate_using_end(const Eigen::VectorXd& new_end) {
  Eigen::VectorXd delta =
      new_end.head(spatial_dim_) - end_state_.head(spatial_dim_);

  start_state_.head(spatial_dim_) += delta;
  end_state_.head(spatial_dim_) = new_end.head(spatial_dim_);
  poly_coeffs_.col(poly_coeffs_.cols() - 1) = start_state_.head(spatial_dim_);
}

void RuckigMotionPrimitive::translate(const Eigen::VectorXd& new_start) {
  MotionPrimitive::translate(new_start);
  compute();
}

Spline MotionPrimitive::add_to_spline(Spline spline, int dim) {
  if (poly_coeffs_.size() == 0) return spline;
  Polynomial poly;
  poly.degree = poly_coeffs_.cols() - 1;
  poly.basis = 0;
  poly.dt = traj_time_;
  if (poly.dt < 0) ROS_ERROR("Negative time in traj");
  poly.start_index = start_index_;
  poly.end_index = end_index_;
  spline.t_total += traj_time_;
  Eigen::VectorXd p = poly_coeffs_.row(dim).reverse();
  // convert between Mike's paramterization and mine
  for (int i = 0; i < p.size(); i++) {
    p[i] *= std::pow(poly.dt, i);
  }
  std::vector<double> pc(p.data(), p.data() + p.size());
  poly.coeffs = pc;
  spline.segments += 1;
  spline.segs.push_back(poly);
  return spline;
}

Spline RuckigMotionPrimitive::add_to_spline(Spline spline, int dim) {
  auto jerk_time_array = ruckig_traj_.get_jerks_and_times();
  std::tuple<float, float, float> state;
  std::get<0>(state) = start_state_[dim];
  std::get<1>(state) = start_state_[dim + spatial_dim_];
  std::get<2>(state) = start_state_[dim + 2 * spatial_dim_];
  for (int seg = 0; seg < 7; seg++) {
    Polynomial poly;
    poly.degree = 3;
    poly.basis = 0;
    poly.dt = jerk_time_array[dim * 2][seg];
    // if (poly.dt == 0) {
    //   continue;
    // }
    float j = jerk_time_array[dim * 2 + 1][seg];
    auto [p, v, a] = state;
    poly.coeffs = {p, v, a / 2, j / 6};
    // convert between Mike's paramterization and mine

    for (int i = 0; i < poly.coeffs.size(); i++) {
      poly.coeffs[i] *= std::pow(poly.dt, i);
    }
    state = ruckig::Profile::integrate(poly.dt, p, v, a, j);
    spline.segments += 1;
    spline.segs.push_back(poly);
    // spline.t_total += poly.dt;
  }
  spline.t_total += ruckig_traj_.duration;
  return spline;
}

Eigen::MatrixXd MotionPrimitive::sample_positions(double step_size) const {
  int num_samples = std::ceil(traj_time_ / step_size) + 1;
  Eigen::VectorXd times =
      Eigen::VectorXd::LinSpaced(num_samples, 0, traj_time_);

  Eigen::MatrixXd result(spatial_dim_, num_samples);

  for (int i = 0; i < times.size(); ++i) {
    result.col(i) = evaluate_primitive(times(i));
  }

  return result;
}

Eigen::VectorXd MotionPrimitive::evaluate_primitive(float t) const {
  Eigen::VectorXd time_multiplier(poly_coeffs_.cols());
  // TODO(laura) could replace with boost::polynomial
  for (int i = 0; i < poly_coeffs_.cols(); ++i) {
    time_multiplier[poly_coeffs_.cols() - i - 1] = std::pow(t, i);
  }
  return poly_coeffs_ * time_multiplier;
}

RuckigMotionPrimitive::RuckigMotionPrimitive(int spatial_dim,
                                             const Eigen::VectorXd& start_state,
                                             const Eigen::VectorXd& end_state,
                                             const Eigen::VectorXd& max_state,
                                             bool recompute)
    : MotionPrimitive(spatial_dim, start_state, end_state, max_state) {
  if (max_state.size() < 4)
    ROS_ERROR("Ruckig MP not valid for control space < 3");
  if (recompute) compute();
}

ETHMotionPrimitive::ETHMotionPrimitive(int spatial_dim,
                                       const Eigen::VectorXd& start_state,
                                       const Eigen::VectorXd& end_state,
                                       const Eigen::VectorXd& max_state,
                                       bool recompute)
    : MotionPrimitive(spatial_dim, start_state, end_state, max_state) {
  if (recompute) compute();
}

void ETHMotionPrimitive::compute(double rho) {
  const int dimension = spatial_dim_;

  // Array for all waypoints and their constrains
  mav_trajectory_generation::Vertex::Vector vertices;

  // // Optimze up to 4th order derivative (SNAP)
  const int derivative_to_optimize =
      mav_trajectory_generation::derivative_order::JERK;

  mav_trajectory_generation::Vertex start(dimension), end(dimension);
  start.addConstraint(mav_trajectory_generation::derivative_order::POSITION,
                      start_state_.head(spatial_dim_));
  start.addConstraint(mav_trajectory_generation::derivative_order::VELOCITY,
                      start_state_.segment(spatial_dim_, spatial_dim_));
  end.addConstraint(mav_trajectory_generation::derivative_order::POSITION,
                    end_state_.head(spatial_dim_));
  end.addConstraint(mav_trajectory_generation::derivative_order::VELOCITY,
                    end_state_.segment(spatial_dim_, spatial_dim_));
  if (start_state_.size() / spatial_dim_ > 2) {
    start.addConstraint(
        mav_trajectory_generation::derivative_order::ACCELERATION,
        start_state_.segment(spatial_dim_ * 2, spatial_dim_));
    end.addConstraint(mav_trajectory_generation::derivative_order::ACCELERATION,
                      end_state_.segment(spatial_dim_ * 2, spatial_dim_));
  }

  vertices.push_back(start);
  vertices.push_back(end);

  // estimate initial segment times
  std::vector<double> segment_times;
  segment_times = estimateSegmentTimes(vertices, max_state_[1], max_state_[2]);
  if (segment_times[0] == 0) {
    segment_times[0] = 1;
  }

  // Set up polynomial solver with default params
  mav_trajectory_generation::NonlinearOptimizationParameters parameters;
  parameters.time_penalty = rho;

  // set up optimization problem
  const int N = 10;
  mav_trajectory_generation::PolynomialOptimizationNonLinear<N> opt(dimension,
                                                                    parameters);
  opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);

  // constrain velocity and acceleration
  opt.addMaximumMagnitudeConstraint(
      mav_trajectory_generation::derivative_order::VELOCITY,
      max_state_[1] + .5);
  opt.addMaximumMagnitudeConstraint(
      mav_trajectory_generation::derivative_order::ACCELERATION,
      max_state_[2] + .5);

  // solve trajectory
  opt.optimize();

  // get trajectory as polynomial parameters
  mav_trajectory_generation::Trajectory trajectory;
  opt.getTrajectory(&(trajectory));
  traj_time_ = trajectory.getSegmentTimes()[0];
  mav_trajectory_generation::Segment seg = trajectory.segments()[0];
  poly_coeffs_.resize(spatial_dim_, N);
  for (int i = 0; i < spatial_dim_; i++) {
    poly_coeffs_.row(i) =
        seg.getPolynomialsRef()[i].getCoefficients(0).reverse();
  }
  // cost_ = opt.getTotalCostWithoutSoftConstraints();
  cost_ = opt.getTotalCostWithSoftConstraints();
  // cost_ = opt.getTotalTimeCost();
  // if (cost_ > 1E6) {
  //   cost_ = -1;
  //   traj_time_ = -1;
  //   poly_coeffs_ = Eigen::MatrixXd();
  // }
}

void RuckigMotionPrimitive::compute() {
  ruckig::Ruckig<3> otg{0.001};
  ruckig::InputParameter<3> input;
  ruckig::OutputParameter<3> output;

  for (int dim = 0; dim < spatial_dim_; dim++) {
    input.max_velocity[dim] = max_state_[1];
    input.max_acceleration[dim] = max_state_[2];
    input.max_jerk[dim] = max_state_[3];
    input.current_position[dim] = start_state_(dim);
    input.current_velocity[dim] = start_state_(spatial_dim_ + dim);
    input.current_acceleration[dim] = start_state_(2 * spatial_dim_ + dim);
    input.target_position[dim] = end_state_(dim);
    input.target_velocity[dim] = end_state_(spatial_dim_ + dim);
    input.target_acceleration[dim] = end_state_(2 * spatial_dim_ + dim);
  }
  if (spatial_dim_ == 2) {
    input.current_position[2] = 0;
    input.current_velocity[2] = 0;
    input.current_acceleration[2] = 0;
    input.target_position[2] = 0;
    input.target_velocity[2] = 0;
    input.target_acceleration[2] = 0;
    input.max_velocity[2] = 1e-2;
    input.max_acceleration[2] = 1e-2;
    input.max_jerk[2] = 1e-2;
  }
  auto result = otg.calculate(input, ruckig_traj_);
  if (result < 0) {
    traj_time_ = -1;
    cost_ = -1;
    ROS_ERROR("Ruckig error %d",
              result);  // TODO(laura) should do more than print
  } else {
    traj_time_ = ruckig_traj_.duration;

    cost_ = traj_time_;
  }
}

Eigen::VectorXd RuckigMotionPrimitive::evaluate_primitive(float t) const {
  std::array<double, 3> position, velocity, acceleration;
  ruckig_traj_.at_time(t, position, velocity, acceleration);
  // Eigen::VectorXd state(3 * spatial_dim_);
  Eigen::VectorXd state(spatial_dim_);
  for (int dim = 0; dim < spatial_dim_; dim++) {
    state[dim] = position[dim];
    // state[spatial_dim_ + dim] = velocity[dim];
    // state[2 * spatial_dim_ + dim] = acceleration[dim];
  }
  return state;
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m) {
  os << "start state: " << m.start_state_.transpose() << "\n";
  os << "end state: " << m.end_state_.transpose() << "\n";
  os << "cost: " << m.cost_ << "\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const MotionPrimitiveGraph& mpg) {
  os << "Vertices:\n" << mpg.vertices_ << "\n";
  return os;
}

std::shared_ptr<MotionPrimitive>
MotionPrimitiveGraph::createMotionPrimitivePtrFromTypeName(
    std::string type_name, int spatial_dim, const Eigen::VectorXd& start_state,
    const Eigen::VectorXd& end_state, const Eigen::VectorXd& max_state) {
  if (type_name == "RuckigMotionPrimitive") {
    return createMotionPrimitivePtr<RuckigMotionPrimitive>(
        spatial_dim, start_state, end_state, max_state);
  } else if (type_name == "PolynomialMotionPrimitive" ||
             type_name == "OptimizationMotionPrimitive") {
    return createMotionPrimitivePtr<PolynomialMotionPrimitive>(
        spatial_dim, start_state, end_state, max_state);
  } else if (type_name == "ETHMotionPrimitive")
    return createMotionPrimitivePtr<ETHMotionPrimitive>(
        spatial_dim, start_state, end_state, max_state);
  else {
    throw;
  }
}

void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph) {
  json_data.at("dispersion").get_to(graph.dispersion_);
  json_data.at("tiling").get_to(graph.tiling_);
  json_data.at("num_dims").get_to(graph.spatial_dim_);
  json_data.at("control_space_q").get_to(graph.control_space_dim_);
  json_data.at("rho").get_to(graph.rho_);
  json_data.at("mp_type").get_to(graph.mp_type_name_);

  graph.state_dim_ = graph.spatial_dim_ * graph.control_space_dim_;
  graph.num_tiles_ = graph.tiling_ ? pow(3, graph.spatial_dim_) : 1;
  auto s = json_data.at("max_state").get<std::vector<double>>();
  graph.max_state_ = Eigen::Map<Eigen::VectorXd>(s.data(), s.size());
  graph.vertices_.resize(json_data["vertices"].size(),
                         graph.spatial_dim_ * graph.control_space_dim_);
  // Convert from json to std::vector to Eigen::VectorXd, maybe could be
  // improved by implementing get for Eigen::VectorXd. For some reason doing
  // this in one line gives the wrong values.
  for (int i = 0; i < json_data.at("vertices").size(); i++) {
    auto x = json_data.at("vertices")[i].get<std::vector<double>>();
    Eigen::Map<Eigen::VectorXd> eigen_vec(x.data(), x.size());
    graph.vertices_.row(i) = eigen_vec;
  }
  graph.edges_.resize(graph.vertices_.rows() * graph.num_tiles_,
                      graph.vertices_.rows());
  for (int i = 0; i < graph.vertices_.rows() * graph.num_tiles_; i++) {
    for (int j = 0; j < graph.vertices_.rows(); j++) {
      auto edge = json_data.at("edges").at(i * graph.vertices_.rows() + j);
      if (edge.size() > 0) {
        auto s = edge.at("start_state").get<std::vector<double>>();
        Eigen::Map<Eigen::VectorXd> start_state(s.data(), s.size());
        auto e = edge.at("end_state").get<std::vector<double>>();
        Eigen::Map<Eigen::VectorXd> end_state(e.data(), e.size());

        Eigen::MatrixXd poly_coeffs;
        if (edge.contains("polys")) {
          poly_coeffs.resize(graph.spatial_dim_, edge.at("polys")[0].size());
          for (int k = 0; k < graph.spatial_dim_; k++) {
            auto x = edge.at("polys")[k].get<std::vector<double>>();
            poly_coeffs.row(k) =
                Eigen::Map<Eigen::VectorXd>(x.data(), x.size());
          }
        }
        auto mp = MotionPrimitiveGraph::createMotionPrimitivePtrFromTypeName(
            graph.mp_type_name_, graph.spatial_dim_, start_state, end_state,
            graph.max_state_);
        mp->populate(edge.at("cost"), edge.at("traj_time"), poly_coeffs, j, i);
        graph.edges_(i, j) = graph.mps_.size();
        graph.mps_.push_back(mp);
      } else {
        graph.edges_(i, j) = -1;  // TODO(laura) make constant
      }
    }
  }
}

MotionPrimitiveGraph read_motion_primitive_graph(const std::string& s) {
  std::ifstream json_file(s);
  nlohmann::json json_data;
  json_file >> json_data;
  return json_data.get<motion_primitives::MotionPrimitiveGraph>();
}

}  // namespace motion_primitives
