from motion_primitives_py import PolynomialMotionPrimitive
import matplotlib.pyplot as plt
import numpy as np
import pytest

# must correspond to object returned from om_fixture otherwise tests invalid
# TODO if om_fixture is changed this will likely break - bad design
@pytest.fixture
def test_pts():
    yield {"unoccupied_valid_pos": np.array([7, 18]),
           "occupied_valid_pos": np.array([6, 11]),
           "invalid_pos": np.array([11, 11]),
           "good_goal": np.array([2, 18]),
           "bad_goal": np.array([7, 7]), }


def test_plot(om_fixture):
    om_fixture.plot()
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def test_is_free_and_valid_position(om_fixture, test_pts):
    assert not om_fixture.is_free_and_valid_position(test_pts["occupied_valid_pos"])
    assert not om_fixture.is_free_and_valid_position(test_pts["invalid_pos"])
    assert om_fixture.is_free_and_valid_position(test_pts["unoccupied_valid_pos"])


def test_is_valid_position(om_fixture, test_pts):
    assert om_fixture.is_valid_position(test_pts["occupied_valid_pos"])
    assert not om_fixture.is_valid_position(test_pts["invalid_pos"])
    assert om_fixture.is_valid_position(test_pts["unoccupied_valid_pos"])


def test_is_mp_collision_free(om_fixture, test_pts):
    start_state = np.pad(test_pts["unoccupied_valid_pos"], (0, 2), 'constant')
    good_goal = np.pad(test_pts["good_goal"], (0, 2), 'constant')
    bad_goal = np.pad(test_pts["bad_goal"], (0, 2), 'constant')
    max_state = 100 * np.ones(4,)
    good_mp = PolynomialMotionPrimitive(start_state, good_goal,
                                        len(om_fixture.dims), max_state)
    bad_mp = PolynomialMotionPrimitive(start_state, bad_goal,
                                       len(om_fixture.dims), max_state)
    assert not om_fixture.is_mp_collision_free(bad_mp)
    assert om_fixture.is_mp_collision_free(good_mp)


if __name__ == "__main__":
    pytest.main(["-v", "--disable-pytest-warnings", __file__])
