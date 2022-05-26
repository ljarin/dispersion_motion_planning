// Copyright 2021 Laura Jarin-Lipschitz

#ifndef INCLUDE_MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H_
#define INCLUDE_MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H_

#include <kr_planning_msgs/Spline.h>

#include <Eigen/Core>
#include <iosfwd>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <ruckig/ruckig.hpp>
#include <string>
#include <vector>

namespace motion_primitives {

class MotionPrimitive {
 public:
  MotionPrimitive() = default;
  MotionPrimitive(int spatial_dim, const Eigen::VectorXd& start_state,
                  const Eigen::VectorXd& end_state,
                  const Eigen::VectorXd& max_state, bool compute = false)
      : spatial_dim_(spatial_dim),
        start_state_(start_state),
        end_state_(end_state),
        max_state_(max_state) {}
  virtual ~MotionPrimitive() {}

  friend std::ostream& operator<<(std::ostream& os, const MotionPrimitive& m);

  int spatial_dim_;
  Eigen::VectorXd start_state_;
  Eigen::VectorXd end_state_;
  Eigen::VectorXd max_state_;

  int start_index_;
  int end_index_;
  double cost_;
  double traj_time_;
  Eigen::MatrixXd poly_coeffs_;

  // Evaluates a motion primitive at a time t and returns a state vector
  virtual Eigen::VectorXd evaluate_primitive(float t) const;

  // Moves the motion primitive to a new position by modifying it's start, end,
  // and polynomial coefficients
  virtual void translate(const Eigen::VectorXd& new_start);
  virtual void translate_using_end(const Eigen::VectorXd& new_end);

  // Samples a motion primitive's position at regular temporal intervals
  // step_size apart.
  // Each row is a position
  virtual Eigen::MatrixXd sample_positions(double step_size = 0.1) const;

  virtual void compute(double rho = 1){};

  virtual void populate(double cost, double traj_time,
                        const Eigen::MatrixXd& poly_coeffs, int start_index,
                        int end_index) {
    cost_ = cost;
    traj_time_ = traj_time;
    poly_coeffs_ = poly_coeffs;
    start_index_ = start_index;
    end_index_ = end_index;
  }

  // Converts and add the dim-th component of the motion primitive to an
  // input Spline message. Lives in the MotionPrimitive class because converting
  // to a polynomial may be different between subclasses (RuckigMotionPrimitive
  // does not store poly_coeffs_ at the moment, just computes them when
  // requested in this function)
  virtual kr_planning_msgs::Spline add_to_spline(
      kr_planning_msgs::Spline spline, int dim);

  // Makes a copy of a shared_ptr to a MotionPrimitive. Smart pointers are
  // needed in e.g. the graph search for the polymorphism of the class to work.
  // Translating the objects inside the graph adjacency matrix caused problems,
  // so we instead make a copy for outputting in the graph search and doing
  // collision checks.
  virtual std::shared_ptr<MotionPrimitive> clone() {
    return std::make_shared<MotionPrimitive>(*this);
  }
};

class PolynomialMotionPrimitive final : public MotionPrimitive {
 public:
  using MotionPrimitive::MotionPrimitive;
};

class RuckigMotionPrimitive final : public MotionPrimitive {
  // TODO(laura) should enforce/warn start_state/end_state must be of dimension
  // 6, makes silent mistakes now
 public:
  RuckigMotionPrimitive() = default;
  RuckigMotionPrimitive(int spatial_dim, const Eigen::VectorXd& start_state,
                        const Eigen::VectorXd& end_state,
                        const Eigen::VectorXd& max_state, bool compute = false);

  ruckig::Trajectory<3> ruckig_traj_;

  Eigen::VectorXd evaluate_primitive(float t) const;
  void translate(const Eigen::VectorXd& new_start);
  void compute();
  kr_planning_msgs::Spline add_to_spline(kr_planning_msgs::Spline spline,
                                          int dim);
  std::shared_ptr<MotionPrimitive> clone() {
    return std::make_shared<RuckigMotionPrimitive>(*this);
  }
};

class ETHMotionPrimitive final : public MotionPrimitive {
 public:
  ETHMotionPrimitive() = default;
  ETHMotionPrimitive(int spatial_dim, const Eigen::VectorXd& start_state,
                     const Eigen::VectorXd& end_state,
                     const Eigen::VectorXd& max_state, bool compute = false);
  void compute(double rho = 1);
};

class MotionPrimitiveGraph {
  friend class GraphSearch;
  friend void from_json(const nlohmann::json& json_data,
                        MotionPrimitiveGraph& graph);
  friend std::ostream& operator<<(std::ostream& out,
                                  const MotionPrimitiveGraph& graph);

 public:
  std::shared_ptr<MotionPrimitive> get_mp_between_indices(
      int i, int j) const noexcept {
    return mps_[edges_(i, j)];
  }

  std::shared_ptr<MotionPrimitive> createMotionPrimitivePtrFromGraph(
      const Eigen::VectorXd& start_state,
      const Eigen::VectorXd& end_state) const {
    return createMotionPrimitivePtrFromTypeName(
        mp_type_name_, spatial_dim_, start_state, end_state, max_state_);
  }

  double rho() const noexcept { return rho_; }
  int spatial_dim() const noexcept { return spatial_dim_; }
  int state_dim() const noexcept { return state_dim_; }
  int control_space_dim() const noexcept { return state_dim_ / spatial_dim_; }
  int num_tiled_states() const noexcept { return edges_.rows(); }
  const auto& max_state() const noexcept { return max_state_; }
  Eigen::MatrixXd vertices() const noexcept { return vertices_; }
  Eigen::ArrayXXi edges() const noexcept { return edges_; }

  bool HasEdge(int i, int j) const noexcept { return edges_(i, j) >= 0; }
  int NormIndex(int i) const noexcept { return std::floor(i / num_tiles_); }

 private:
  std::vector<std::shared_ptr<MotionPrimitive>>
      mps_;                // TODO(laura) maybe should be unique_ptr
  Eigen::ArrayXXi edges_;  // indexing is kind of counterinuitive (end index =
                           // row, start index = column)
  Eigen::MatrixXd vertices_;
  Eigen::VectorXd max_state_;

  double dispersion_;
  double rho_ = 1;  // TODO(laura) decide about using rho in graph search
                    // convention (has to do with time optimal vs. LQMT cost)
  int spatial_dim_;
  int control_space_dim_;
  int state_dim_;
  int num_tiles_;
  bool tiling_;
  std::string mp_type_name_;

  template <typename T>
  static std::shared_ptr<MotionPrimitive> createMotionPrimitivePtr(
      int spatial_dim, const Eigen::VectorXd& start_state,
      const Eigen::VectorXd& end_state, const Eigen::VectorXd& max_state) {
    return std::make_shared<T>(spatial_dim, start_state, end_state, max_state);
  }

  static std::shared_ptr<MotionPrimitive> createMotionPrimitivePtrFromTypeName(
      std::string type_name, int spatial_dim,
      const Eigen::VectorXd& start_state, const Eigen::VectorXd& end_state,
      const Eigen::VectorXd& max_state);
};

// Overrides a function from nlohmann::json to convert a json file into a
// MotionPrimitiveGraph object.
void from_json(const nlohmann::json& json_data, MotionPrimitiveGraph& graph);
// Creates the intermediate json objects to convert from a file location to a
// MotionPrimitiveGraph.
MotionPrimitiveGraph read_motion_primitive_graph(const std::string& s);

template <typename T>
std::ostream& operator<<(std::ostream& output, std::vector<T> const& values) {
  for (auto const& value : values) {
    output << value << "\n";
  }
  return output;
}

}  // namespace motion_primitives

#endif  // INCLUDE_MOTION_PRIMITIVES_MOTION_PRIMITIVE_GRAPH_H_
