import numpy as np
import pytest
from copy import deepcopy

# TODO test graph search termination
def test_search(search_fixture):
    search_fixture.run_graph_search()
    assert search_fixture.succeeded is True
    search_fixture.make_graph_search_animation()


def test_fail_search(fail_search_fixture):
    fail_search_fixture.run_graph_search()
    assert fail_search_fixture.succeeded is False

# TODO(laura) fix
# def test_simple_graph(simple_search_fixture):
#     simple_search_fixture.run_graph_search()
# const auto path = gs.Search({.start_state = start,
#                             .goal_state = goal,
#                             .distance_threshold = 0.001,
#                             .parallel_expand = true,
#                             .using_ros = false});

# float path_cost;
# for (auto seg : path) {
# path_cost += seg.cost;
# }
# EXPECT_EQ(path_cost, 2);


if __name__ == '__main__':
    pytest.main(["-v", "--disable-pytest-warnings", __file__])
