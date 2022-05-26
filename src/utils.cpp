// Copyright 2021 Laura Jarin-Lipschitz
#include "motion_primitives/utils.h"

#include <geometry_msgs/Point.h>

namespace motion_primitives {

using geometry_msgs::Point;
using kr_planning_msgs::Polynomial;
using kr_planning_msgs::Primitive;
using kr_planning_msgs::Spline;
using kr_planning_msgs::SplineTrajectory;
using kr_planning_msgs::Trajectory;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

Trajectory path_to_traj_msg(
    const std::vector<std::shared_ptr<MotionPrimitive>>& mps,
    const std_msgs::Header& header, float z_height) {
  if (mps.empty()) return {};

  int spatial_dim = mps[0]->spatial_dim_;
  Eigen::ArrayXXd pc_resized(spatial_dim, 6);
  Eigen::ArrayXXd coeff_multiplier(pc_resized.rows(), pc_resized.cols());
  Trajectory trajectory;

  trajectory.header = header;
  trajectory.primitives.reserve(mps.size());
  //  trajectory.header.stamp = ros::Time::now();
  //  trajectory.header.frame_id = voxel_map_.header.frame_id;

  for (int i = 0; i < pc_resized.rows(); ++i) {
    // These hardcoded coefficients come from how
    // kr_planning_msgs::Primitive/MPL defines polynomial trajectories
    coeff_multiplier.row(i) << 120, 24, 6, 2, 1, 1;
  }

  for (const auto& mp : mps) {
    if (mp->poly_coeffs_.size() == 0) break;

    Primitive primitive;
    pc_resized.block(0, pc_resized.cols() - mp->poly_coeffs_.cols(),
                     pc_resized.rows(), mp->poly_coeffs_.cols()) =
        mp->poly_coeffs_;

    pc_resized *= coeff_multiplier;
    for (int i = 0; i < pc_resized.cols(); i++) {
      primitive.cx.push_back(pc_resized(0, i));
      primitive.cy.push_back(pc_resized(1, i));
      if (spatial_dim > 2) {
        primitive.cz.push_back(pc_resized(2, i));
      } else {
        primitive.cz.push_back(0.);
        if (i == pc_resized.cols() - 1) primitive.cz.back() = z_height;
      }
      primitive.cyaw.push_back(0.);
    }
    primitive.t = mp->traj_time_;
    trajectory.primitives.push_back(primitive);
  }
  return trajectory;
}

SplineTrajectory path_to_spline_traj_msg(
    const std::vector<std::shared_ptr<MotionPrimitive>>& mps,
    const std_msgs::Header& header, float z_height) {
  if (mps.empty()) return {};

  SplineTrajectory spline_traj;
  spline_traj.header = header;
  spline_traj.dimensions = 3;

  for (int dim = 0; dim < 3; dim++) {
    Spline spline;
    if (mps[0]->spatial_dim_ != dim) {
      for (const auto& mp : mps) {
        spline = mp->add_to_spline(spline, dim);
      }
    } else {
      spline.segments = spline_traj.data[0].segments;
      spline.t_total = spline_traj.data[0].t_total;
      for (int i = 0; i < spline.segments; i++) {
        Polynomial poly;
        poly.degree = spline_traj.data[0].segs[0].degree;
        for (int j = 0; j < poly.degree + 1; j++) {
          poly.coeffs.push_back(0.);
        }
        poly.coeffs[0] = {z_height};
        poly.dt = spline_traj.data[0].segs[i].dt;
        spline.segs.push_back(poly);
      }
    }
    spline_traj.data.push_back(spline);
  }
  return spline_traj;
}

MarkerArray StatesToMarkerArray(const std::vector<Eigen::VectorXd>& states,
                                int spatial_dim, const std_msgs::Header& header,
                                double scale, bool show_vel) {
  MarkerArray marray;
  marray.markers.reserve(2);

  // end point of each mps, put them in the same marker as a sphere list to
  // speed up rendering
  Marker m_pos;
  m_pos.id = 0;
  m_pos.ns = "pos";
  m_pos.header = header;
  m_pos.color.b = 1.0;
  m_pos.color.a = 0.5;
  m_pos.type = Marker::SPHERE_LIST;
  m_pos.scale.x = m_pos.scale.y = m_pos.scale.z = scale;
  m_pos.pose.orientation.w = 1.0;
  for (const auto& state : states) {
    Point p;
    p.x = state.x();
    p.y = state.y();
    p.z = spatial_dim == 3 ? state.z() : 0;
    m_pos.points.push_back(p);
  }
  marray.markers.push_back(std::move(m_pos));

  if (show_vel) {
    Marker m_vel;
    m_vel.id = 0;
    m_vel.ns = "vel";
    m_vel.header = header;
    m_vel.color.b = 1.0;
    m_vel.color.a = 0.5;
    m_vel.type = Marker::LINE_LIST;
    m_vel.scale.x = m_pos.scale.y = m_pos.scale.z = scale / 4.0;
    m_vel.pose.orientation.w = 1.0;
    for (const auto& state : states) {
      auto pos = state.head(spatial_dim);
      Point p1;
      p1.x = pos[0];
      p1.y = pos[1];
      p1.z = spatial_dim == 3 ? state[2] : 0;
      Point p2 = p1;
      auto vel = state.tail(spatial_dim) / 4.0;
      p2.x += vel[0];
      p2.y += vel[1];
      p2.z += spatial_dim == 3 ? vel[2] : 0;
      m_vel.points.push_back(p1);
      m_vel.points.push_back(p2);
    }
    marray.markers.push_back(std::move(m_vel));
  }

  return marray;
}

}  // namespace motion_primitives
