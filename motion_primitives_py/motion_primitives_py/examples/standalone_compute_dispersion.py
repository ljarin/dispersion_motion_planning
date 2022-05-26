import numpy as np
import matplotlib.pyplot as plt

"""
Standalone (does not use this package at all) example of numerically computing Euclidean dispersion
"""


def uniform_state_set(num_dims, bounds, resolution, random=False):
    """
    Return a uniform Cartesian sampling over vector bounds with vector resolution.
    Input:
        bounds, (N) bounds over N dimensions (assumes symmetric positive and negative values)
        resolution, (N,) resolution over N dimensions
    Output:
        pts, (M,N) set of M points sampled in N dimensions
    """
    assert len(bounds) == len(resolution)
    independent = []
    bounds = np.asarray(bounds)
    for (a, r) in zip(bounds, resolution):
        for _ in range(num_dims):
            if random:
                independent.append(np.random.rand((np.ceil(a/r+1).astype(int)))*2*a-a)
            else:
                if r != np.inf:
                    independent.append(np.concatenate([np.flip(-np.arange(0, a+.00001, r)[1:]), np.arange(0, a+.00001, r)]))
                else:
                    independent.append(0)  # if the requested resolution is infinity, just return 0
    joint = np.meshgrid(*independent)
    pts = np.stack([j.ravel() for j in joint], axis=-1)
    return pts


def dispersion_distance_fn_simple_norm(start_pts, end_pts):
    """
    Compute the norm of the Euclidean distance from every start_pt to every end_pt
    """
    score = np.linalg.norm(start_pts[:, np.newaxis]-end_pts, axis=2)
    return score


def compute_dispersion(samples, max_state, resolution):
    # densely sample the state space (checking as many points in the state space as possible for how close they are to one of your samples)
    dense_sampling = uniform_state_set(samples.shape[1], max_state, resolution, random=False)
    # compute the distances from every 'dense' sample to your sample set
    score = dispersion_distance_fn_simple_norm(dense_sampling, samples)
    # for every dense pt, keep only the shortest distance from it to the samples
    min_score = np.nanmin(score, axis=1)
    # Take the max of the above array, yielding the dense sample that it furthest from a member of the sample set
    index = np.argmax(min_score)
    dispersion = min_score[index]

    fig, ax = plt.subplots()
    ax.plot(samples[:, 0], samples[:, 1], 'og')
    circle = plt.Circle(dense_sampling[index, :2], dispersion, color='b', fill=False, )
    ax.add_artist(circle)

    return dispersion


if __name__ == "__main__":
    import itertools
    samples = np.array(list(itertools.product(np.arange(-10, 10), np.arange(-10, 10))))
    compute_dispersion(samples, [9], [.1])
    plt.show()
