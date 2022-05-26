cmake_minimum_required(VERSION 3.11)
project(motion_primitives)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# import nlohmann json library, must be first pulled with vcs tool
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_MultipleHeaders ON CACHE INTERNAL "")
add_subdirectory(external/json)

set(BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(BUILD_TESTS OFF CACHE INTERNAL "")
add_subdirectory(external/ruckig)

# Find catkin macros and libraries if COMPONENTS list like find_package(catkin
# REQUIRED COMPONENTS xyz) is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp kr_planning_msgs rosbag actionlib mav_trajectory_generation pybind11_catkin cmake_modules)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer)
find_package(TBB REQUIRED)

catkin_package(
  INCLUDE_DIRS include 
  LIBRARIES ${PROJECT_NAME} 
  DEPENDS TBB
  CATKIN_DEPENDS roscpp kr_planning_msgs rosbag actionlib mav_trajectory_generation
)

include_directories(
  include  
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
# ##############################################################################
# Build ##
# ##############################################################################

# Specify libraries to link a library or executable target against
add_library(${PROJECT_NAME} src/motion_primitive_graph.cpp src/graph_search.cpp src/utils.cpp)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  nlohmann_json::nlohmann_json
  ${Boost_LIBRARIES}
  ${TBB_LIBRARIES} 
  ruckig
  )

add_executable(${PROJECT_NAME}_graph_search src/graph_search_node.cpp)
target_link_libraries(${PROJECT_NAME}_graph_search PRIVATE ${PROJECT_NAME})

add_executable(${PROJECT_NAME}_action_server src/graph_search_action_server.cpp)
target_link_libraries(${PROJECT_NAME}_action_server PRIVATE ${PROJECT_NAME})

# Mark executables for installation See
# http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS ${PROJECT_NAME}_graph_search
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# Mark libraries for installation See
# http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE)

add_subdirectory(tests)

pybind_add_module(motion_primitives_cpp MODULE src/python.cpp)
target_link_libraries(motion_primitives_cpp PUBLIC ${PROJECT_NAME})

install(TARGETS motion_primitives_cpp
  LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)