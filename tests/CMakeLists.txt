# see https://catkin-tools.readthedocs.io/en/latest/verbs/catkin_build.html for details
# or just do `catkin run_tests motion_primitives`
find_package(GTest REQUIRED)
catkin_add_gtest(test_motion_primitive test_motion_primitive.cpp)
target_link_libraries(test_motion_primitive ${PROJECT_NAME} GTest::Main)

catkin_add_gtest(test_graph_search test_graph_search.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_graph_search ${PROJECT_NAME} GTest::Main)

catkin_add_gtest(test_graph test_motion_primitive_graph.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test_graph ${PROJECT_NAME} GTest::Main)

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_motion_primitives_py.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
