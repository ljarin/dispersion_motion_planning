import pytest
import numpy as np
from motion_primitives_py import MotionPrimitiveLattice
import tempfile
import os


def test_lattice_save_load(lattice_fixture):
    with tempfile.TemporaryDirectory() as td:
        f_name = os.path.join(td, 'temp')
        with open(f_name, 'w') as tf:
            lattice_fixture.save(tf.name)
        with open(f_name) as tf:
            mpl2 = MotionPrimitiveLattice.load(tf.name)
    assert((lattice_fixture.vertices == mpl2.vertices).all())
    assert((lattice_fixture.edges == mpl2.edges).all())
    assert(lattice_fixture.control_space_q == mpl2.control_space_q)
    assert(lattice_fixture.num_dims == mpl2.num_dims)
    assert((lattice_fixture.max_state == mpl2.max_state).all())
    assert(lattice_fixture.motion_primitive_type == mpl2.motion_primitive_type)
    assert(lattice_fixture.num_tiles == mpl2.num_tiles)


if __name__ == '__main__':
    pytest.main(["-v", "--disable-pytest-warnings", __file__])
