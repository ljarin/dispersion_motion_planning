
def reduce_graph_degree(mpl):
    pts, independent = mpl.uniform_state_set(mpl.max_state[:mpl.control_space_q], mpl.resolution[:mpl.control_space_q], random=False)

    paths = np.empty((len(mpl.edges), len(mpl.vertices)), dtype=object)
    for i in range(len(mpl.edges)):
        for j in range(len(mpl.vertices)):
            if i != j:
                paths[i, j] = bfs(mpl, i, j)

    counter = 0
    for i in range(len(mpl.edges)):
        for j in range(len(mpl.vertices)):
            if paths[i, j] is not None:
                if len(paths[i, j]) > 1:
                    counter +=1
                    candidate_path = np.argmax([cost for path, cost in paths[i, j]])
                    # for k, (path, cost) in enumerate(paths[i,j]):
                        # if k!= candidate_path:
                            # for edge in path:
                                # print(independent)
                                # print(np.where(independent==np.array(edge.start_state)))
                                # print(edge.start_state)
                                # print(edge.end_state)
    print(counter)
    print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None]))

def bfs(mpl, i, j):
    # bfs https://pythoninwonderland.wordpress.com/2017/03/18/how-to-implement-breadth-first-search-in-python/
    original_edge = mpl.edges[i, j]
    if original_edge is None:
        return None
    explored = []
    queue = [[original_edge]]
    paths = [([original_edge], original_edge.cost)]
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node not in explored:
            # neighbors = mpl.edges[:, node]
            neighbors = mpl.get_neighbor_mps(j)
            # go through all neighbor nodes, construct a new path and
            # push it into the queue
            for neighbor in neighbors:
                if neighbor is not None and not (neighbor.start_state == neighbor.end_state).all():
                    new_path = list(path)
                    new_path.append(neighbor)
                    new_path_cost = sum([m.cost for m in new_path])
                    if new_path_cost < 2*mpl.dispersion:
                        queue.append(new_path)
                        # return path if neighbor is goal
                        if(neighbor.end_state == original_edge.end_state).all():
                            paths.append((new_path, new_path_cost))

            # mark node as explored
            explored.append(node)
    return paths


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from motion_primitives_py import *

    ## Create graph
    motion_primitive_type = PolynomialMotionPrimitive
    control_space_q = 2
    num_dims = 2
    max_state = [.51, 1.51, 15, 100]
    mp_subclass_specific_data = {'iterative_bvp_dt': .05, 'iterative_bvp_max_t': 2}
    resolution = [.21, .51]
    tiling = True
    check_backwards_dispersion = False
    mpl = MotionPrimitiveLattice(control_space_q, num_dims, max_state, motion_primitive_type, tiling, False, mp_subclass_specific_data)
    mpl.compute_min_dispersion_space(
        num_output_pts=20, resolution=resolution, check_backwards_dispersion=check_backwards_dispersion)
    mpl.limit_connections(2*mpl.dispersion) # limit connectivity
    mpl.save("lattice_test.json") # save graph to json

    # Reduce graph degree
    # mpl = MotionPrimitiveLattice.load("lattice_test.json", True) # can load a saved graph instead of recomputing it
    print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices)) # how many edges per vertex

    reduce_graph_degree(mpl)
    print(sum([1 for i in np.nditer(mpl.edges, ['refs_ok']) if i != None])/len(mpl.vertices))
