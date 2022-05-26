#include <motion_primitives/graph_search.h>
#include <motion_primitives/motion_primitive_graph.h>
#include <motion_primitives/utils.h>
#include <kr_planning_msgs/VoxelMap.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace motion_primitives;

PYBIND11_MODULE(motion_primitives_cpp, m) {
  m.doc() = "motion_primitives python bindings";

  py::class_<GraphSearch::Option>(m, "Option")
      .def(py::init<>())
      .def_readwrite("start_state", &GraphSearch::Option::start_state)
      .def_readwrite("goal_state", &GraphSearch::Option::goal_state)
      .def_readwrite("distance_threshold",
                     &GraphSearch::Option::distance_threshold)
      .def_readwrite("parallel_expand", &GraphSearch::Option::parallel_expand)
      .def_readwrite("heuristic", &GraphSearch::Option::heuristic)
      .def_readwrite("access_graph", &GraphSearch::Option::access_graph)
      .def_readwrite("using_ros", &GraphSearch::Option::using_ros);

  py::class_<MotionPrimitiveGraph>(m, "MotionPrimitiveGraph").def(py::init<>());
  py::class_<::kr_planning_msgs::VoxelMap>(m, "VoxelMap")
      .def(py::init<>())
      .def_readwrite("resolution", &::kr_planning_msgs::VoxelMap::resolution);

  py::class_<GraphSearch>(m, "GraphSearch")
      .def(py::init<const MotionPrimitiveGraph&,
                    const kr_planning_msgs::VoxelMap&,
                    const GraphSearch::Option&>())
      .def("Search",
           [](GraphSearch& gs) {
             auto x = gs.Search();
             return x.second;
           })
      .def("num_visited", [](const GraphSearch& gs) {
        return (int)gs.GetVisitedStates().size();
      });

  py::class_<GraphSearch::Node>(m, "GraphSearch::Node")
      .def(py::init<>())
      .def_readonly("motion_cost", &GraphSearch::Node::motion_cost);
  py::class_<MotionPrimitive>(m, "MotionPrimitive").def(py::init<>());
  py::class_<ETHMotionPrimitive>(m, "ETHMotionPrimitive").def(py::init<>());

  m.def("read_motion_primitive_graph", &read_motion_primitive_graph);

  m.def("read_bag", &read_bag<kr_planning_msgs::VoxelMap>);
}
