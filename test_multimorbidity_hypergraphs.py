
import pytest
import multimorbidity_hypergraphs as hgt
import numpy as np
import pandas as pd
import numba
import statsmodels.stats.proportion as smsp
import scipy.stats as sst

def test_instantiated():
    """
    Tests the instantiation of the hypergraph object.

    Pretty simple test as all internal state is set to None.
    """

    h = hgt.Hypergraph()

    assert h.incidence_matrix is None
    assert h.edge_weights is None
    assert h.edge_weights_ci is None
    assert h.node_weights is None
    assert h.node_weights_ci is None
    assert h.edge_list is None
    assert h.node_list is None


def test_build_hypergraph_edge_weights():
    """
    Test the calculation of the edges weights in the construction of a
    hypergraph with a very simple dataset

    The expected edge weights have been calculated by hand and are stored in
    exp_edge_weights.
    """

    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]
    ])

    exp_edge_weights = {
        ('disease_0', 'disease_1'): 1/2,
        ('disease_0', 'disease_2'): 2/2,
        ('disease_1', 'disease_2'): 2/3,
        ('disease_0', 'disease_3'): 1/2,
        ('disease_1', 'disease_3'): 3/3,
        ('disease_2', 'disease_3'): 2/3,
        ('disease_0', 'disease_1', 'disease_2'): 1/2,
        ('disease_0', 'disease_1', 'disease_3'): 1/2,
        ('disease_0', 'disease_2', 'disease_3'): 1/2,
        ('disease_1', 'disease_2', 'disease_3'): 2/3,
    }

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    # make sure there are the right number of sets / weights
    assert len(h.edge_weights) == len(exp_edge_weights.values())

    # check each weight
    for k in exp_edge_weights:
        assert h.edge_weights[h.edge_list.index(k)] == exp_edge_weights[k]


def test_build_hypergraph_edge_weights_zero_sets():
    """
    Test that edges with zero weight are correctly discarded.
    """

    data = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])  # there is no one with disease 0 and disease 1

    exp_edge_weights = {
        ('disease_0', 'disease_1'): 0,
        ('disease_0', 'disease_2'): 1/1,
        ('disease_1', 'disease_2'): 2/3,
    }

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    # zero weights are discarded. There is EXACTLY one zero weight in this system
    assert len(h.edge_weights) + 1 == len(exp_edge_weights.values())

    # check each non-zero weight
    for k in exp_edge_weights:
        if exp_edge_weights[k] > 0:
            assert h.edge_weights[h.edge_list.index(k)] == exp_edge_weights[k]

    assert len(h.edge_weights) == len(h.edge_weights_ci)
    assert len(h.edge_weights) == len(h.edge_weights_pop)


def test_build_hypergraph_edge_weights_zero_sets_custom_weights():
    """
    Test that edges with zero weight are correctly discarded when using a custom weight function.
    """
    
    
    @numba.jit(
        nopython=True,
        nogil=True,
        fastmath=True,
    )
    def unit_weights(data, inds):
        """
        This function returns a 1.0 divided by a number passed in as an optional
        argument.
        """
        if len(inds) == 3:
            return 0.0, 0.0, 0.0
        return 1.0, 0.0, 0.0
        
    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]
    ])

    exp_edge_weights = {
        ('disease_0', 'disease_1'): 1,
        ('disease_0', 'disease_2'): 1,
        ('disease_1', 'disease_2'): 1,
        ('disease_0', 'disease_3'): 1,
        ('disease_1', 'disease_3'): 1,
        ('disease_2', 'disease_3'): 1,
        #('disease_0', 'disease_1', 'disease_2'): 0,
        #('disease_0', 'disease_1', 'disease_3'): 0,
        #('disease_0', 'disease_2', 'disease_3'): 0,
        #('disease_1', 'disease_2', 'disease_3'): 0,
    }

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd, unit_weights)

    # make sure there are the right number of sets / weights
    assert len(h.edge_weights) == len(exp_edge_weights.values())

    # check each non-zero weight
    for k in exp_edge_weights:
        if exp_edge_weights[k] > 0:
            assert h.edge_weights[h.edge_list.index(k)] == exp_edge_weights[k]

    assert len(h.edge_weights) == len(h.edge_weights_ci)
    assert len(h.edge_weights) == len(h.edge_weights_pop)




def test_build_hypergraph_node_weights():
    """
    Test that node weights (crude prevalence) are correctly calculated.

    The expected node weights have been calculated by hand and are stored in
    exp_node_weights.
    """

    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]
    ])

    exp_node_weights = {
        'disease_0': 2/5,
        'disease_1': 4/5,
        'disease_2': 3/5,
        'disease_3': 3/5,
    }

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    for k in exp_node_weights:
        # This rounding error is caused by fast_math being set to true in the
        # numba JIT decorator.
        assert np.abs(h.node_weights[h.node_list.index(k)] - exp_node_weights[k]) < 1e-15


def test_build_hypergraph_incidence_matrix():
    """
    Test that incidence matrix is correctly calculated.

    The expected incidence matrix is stored in exp_incidence_matrix and needs
    to have it's rows reordered as the edge list is shuffled in h.compute_hypergraph()
    to improve threading performance.
    """

    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1]
    ])

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    exp_edge_list = [
        ('disease_0', 'disease_1'),
        ('disease_0', 'disease_2'),
        ('disease_1', 'disease_2'),
        ('disease_0', 'disease_3'),
        ('disease_1', 'disease_3'),
        ('disease_2', 'disease_3'),
        ('disease_0', 'disease_1', 'disease_2'),
        ('disease_0', 'disease_1', 'disease_3'),
        ('disease_0', 'disease_2', 'disease_3'),
        ('disease_1', 'disease_2', 'disease_3'),
    ]

    exp_incidence_matrix = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ])

    # the edge list is randomly shuffled for threading.
    inds = [exp_edge_list.index(k) for k in h.edge_list]
    exp_incidence_matrix = exp_incidence_matrix[inds, :]

    assert (exp_incidence_matrix == h.incidence_matrix).all()


def test_calculate_EVC_standard_hypergraph():
    """
    Test that the eigenvector centrality of the standard hypergraph
    (centrality of the nodes) is calculated correctly.
    """

    n_people = 5000
    n_diseases = 10
    tolerance = 1e-6

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    # calculate the adjacency matrix from the incidence matrix and weights
    # computed by h.compute_hypergraph() tested above.

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    adjacency_matrix = np.dot(
        h.incidence_matrix.T,
        np.dot(
            np.diag(h.edge_weights),
            h.incidence_matrix
        )
    )
    np.fill_diagonal(adjacency_matrix, 0.0)
    np_e_vals, np_e_vecs = np.linalg.eigh(adjacency_matrix)

    exp_eval = np.max(np_e_vals)
    exp_evec = np_e_vecs[:, exp_eval == np_e_vals].reshape(-1)
    exp_evec = exp_evec / np.sqrt(np.dot(exp_evec, exp_evec))

    # The expected eigenvector can sometimes be all negative elements, for
    # what I assume are numerical reasons. They should always be either all
    # positive or all negative (i.e. up to an overal scaling of -1).
    assert (exp_evec > 0).all() | (exp_evec < 0).all()
    exp_evec = np.abs(exp_evec)

    e_vec, e_vec_err = h.eigenvector_centrality(tolerance=tolerance)

    # eigenvectors are defined up to a scaling, so normalise such that it is a unit vector.
    e_vec = e_vec / np.sqrt(np.dot(e_vec, e_vec))

    # there is some numerical uncertainty in these calculations
    #assert np.abs(exp_eval - e_val) ** 2 < tolerance
    assert (np.abs(exp_evec - e_vec) < tolerance).all()


def test_weighted_resultant_EVC_standard_hypergraph():
    """
    Test that the eigenvector centrality of the standard hypergraph
    (centrality of the nodes, with both node and edge weights included)
    is calculated correctly.
    """

    n_people = 5000
    n_diseases = 10
    tolerance = 1e-6

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    # calculate the adjacency matrix from the incidence matrix and weights
    # computed by h.compute_hypergraph() tested above.

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)


    adjacency_matrix = np.dot(
        np.dot(
            np.diag(np.sqrt(h.node_weights)),
            np.dot(
                h.incidence_matrix.T,
                np.dot(
                    np.diag(h.edge_weights),
                    h.incidence_matrix
                )
            )
        ),
        np.diag(np.sqrt(h.node_weights))
    )
    np.fill_diagonal(adjacency_matrix, 0.0)
    np_e_vals, np_e_vecs = np.linalg.eigh(adjacency_matrix)

    exp_eval = np.max(np_e_vals)
    exp_evec = np_e_vecs[:, exp_eval == np_e_vals].reshape(-1)
    exp_evec = exp_evec / np.sqrt(np.dot(exp_evec, exp_evec))

    # The expected eigenvector can sometimes be all negative elements, for
    # what I assume are numerical reasons. They should always be either all
    # positive or all negative (i.e. up to an overal scaling of -1).
    assert (exp_evec > 0).all() | (exp_evec < 0).all()
    exp_evec = np.abs(exp_evec)

    e_vec, e_vec_err = h.eigenvector_centrality(tolerance=tolerance, weighted_resultant=True)

    # eigenvectors are defined up to a scaling, so normalise such that it is a unit vector.
    e_vec = e_vec / np.sqrt(np.dot(e_vec, e_vec))

    # there is some numerical uncertainty in these calculations
    #assert np.abs(exp_eval - e_val) ** 2 < tolerance
    assert (np.abs(exp_evec - e_vec) < tolerance).all()


def test_calculate_EVC_dual_hypergraph():
    """
    Test that the eigenvector centrality of the dual hypergraph
    (centrality of the edges) is calculated correctly.
    """

    n_people = 5000
    n_diseases = 10
    tolerance = 1e-6

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    adjacency_matrix = np.dot(
        h.incidence_matrix,
        np.dot(
            np.diag(h.node_weights),
            h.incidence_matrix.T
        )
    )
    np.fill_diagonal(adjacency_matrix, 0.0)
    np_e_vals, np_e_vecs = np.linalg.eigh(adjacency_matrix)

    exp_eval = np.max(np_e_vals)
    exp_evec = np_e_vecs[:, exp_eval == np_e_vals].reshape(-1)
    exp_evec = exp_evec / np.sqrt(np.dot(exp_evec, exp_evec))

    assert (exp_evec > 0).all() | (exp_evec < 0).all()
    exp_evec = np.abs(exp_evec)

    e_vec, e_vec_err = h.eigenvector_centrality(
        rep="dual",
        tolerance=tolerance
    )
    # eigenvectors are defined up to a scaling, so normalise such that it is a unit vector.
    e_vec = e_vec / np.sqrt(np.dot(e_vec, e_vec))

    # there is some numerical uncertainty in these calculations
    #assert (exp_eval - e_val) ** 2 < tolerance
    assert (np.abs(exp_evec - e_vec) < tolerance).all()


def test_weighted_resultant_EVC_dual_hypergraph():
    """
    Test that the eigenvector centrality of the dual hypergraph
    (centrality of the edges, with both node and edge weights included)
    is calculated correctly.
    """

    n_people = 5000
    n_diseases = 10
    tolerance = 1e-6

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    adjacency_matrix = np.dot(
        np.dot(
            np.diag(np.sqrt(h.edge_weights)),
            np.dot(
                h.incidence_matrix,
                np.dot(
                    np.diag(h.node_weights),
                    h.incidence_matrix.T
                )
            )
        ),
        np.diag(np.sqrt(h.edge_weights))
    )

    np.fill_diagonal(adjacency_matrix, 0.0)
    np_e_vals, np_e_vecs = np.linalg.eigh(adjacency_matrix)

    exp_eval = np.max(np_e_vals)
    exp_evec = np_e_vecs[:, exp_eval == np_e_vals].reshape(-1)
    exp_evec = exp_evec / np.sqrt(np.dot(exp_evec, exp_evec))

    assert (exp_evec > 0).all() | (exp_evec < 0).all()
    exp_evec = np.abs(exp_evec)

    e_vec, e_vec_err = h.eigenvector_centrality(
        rep="dual",
        weighted_resultant=True,
        tolerance=tolerance
    )
    # eigenvectors are defined up to a scaling, so normalise such that it is a unit vector.
    e_vec = e_vec / np.sqrt(np.dot(e_vec, e_vec))

    # there is some numerical uncertainty in these calculations
    #assert (exp_eval - e_val) ** 2 < tolerance
    assert (np.abs(exp_evec - e_vec) < tolerance).all()


def test_calculate_EVC_bipartite_hypergraph():
    """
    Test that the eigenvector centrality of the bipartite hypergraph
    (centrality of the nodes and the edges) is calculated correctly.
    """

    n_people = 5000
    n_diseases = 10
    tolerance = 1e-6

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    total_elems = len(h.edge_list) + len(h.node_list)

    adjacency_matrix = np.zeros((total_elems, total_elems))

    adjacency_matrix[len(h.node_list):total_elems, 0:len(h.node_list)] = np.dot(
        h.incidence_matrix.T,
        np.diag(h.edge_weights)
    ).T
    adjacency_matrix[0:len(h.node_list), len(h.node_list):total_elems] = np.dot(
        h.incidence_matrix.T,
        np.diag(h.edge_weights)
    )

    np_e_vals, np_e_vecs = np.linalg.eigh(adjacency_matrix)

    exp_eval = np.max(np_e_vals)
    exp_evec = np_e_vecs[:, exp_eval == np_e_vals].reshape(-1)
    exp_evec = exp_evec / np.sqrt(np.dot(exp_evec, exp_evec))

    assert (exp_evec > 0).all() | (exp_evec < 0).all()
    exp_evec = np.abs(exp_evec)

    e_vec, e_vec_err = h.eigenvector_centrality(
        rep="bipartite",
        tolerance=tolerance
    )

    e_vec = e_vec / np.sqrt(np.dot(e_vec, e_vec))

    # I don't completely understand how the tolerance relates to the error.
    # There is probably some addtional uncertainty coming from the fast_math
    # approximations that numba is using, and this bipartite adjacency matrix
    # is contructed in a really ad hoc way. The differences between expectation
    # and the module code is
    # a) consistent and
    # b) small compared to the eigenvector elements ( O(0.01%) ).
    #assert (exp_eval - e_val) ** 2 < tolerance
    assert (np.abs(exp_evec - e_vec) ** 2 < tolerance).all()


def test_EVC_exception_raised():
    """
    Tests that an exception is raised when an incorrect representation
    string is used
    """

    h = hgt.Hypergraph()
    with pytest.raises(Exception):
        h.eigenvector_centrality(rep="oh no!")


def test_degree_centrality_weighted():

    """
    Test the calculation of the weighted degree centrality for a hypergraph
    """

    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1]
    ])

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)
    edge_node_list = [item for sublist in h.edge_list for item in sublist]

    exp_degree_centrality = []

    for node in h.node_list:
        dc = 0.0
        for edge, weight in zip(h.edge_list, h.edge_weights):
            for edge_node in edge:
                if node == edge_node:
                    dc += weight
        exp_degree_centrality.append(dc)


    degree_centrality = h.degree_centrality()

    for (act, exp) in zip(exp_degree_centrality, degree_centrality):
        assert act == exp


def test_edge_degree_centrality_weighted():

    """
    Test the calculation of the weighted degree centrality for a dual hypergraph
    """

    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1]
    ])

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)
    edge_node_list = [item for sublist in h.edge_list for item in sublist]

    exp_degree_centrality = []

    for edge in h.edge_list:
        dc = 0.0
        for node in edge:
            dc += h.node_weights[h.node_list.index(node)]
        exp_degree_centrality.append(dc)

    degree_centrality = h.degree_centrality(rep="dual")

    for (act, exp) in zip(exp_degree_centrality, degree_centrality):
        assert act == exp



def test_degree_centrality_unweighted():

    """
    Test the calculation of the degree centrality for a hypergraph
    with unit weights
    """

    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1]
    ])

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)
    edge_node_list = [item for sublist in h.edge_list for item in sublist]

    exp_degree_centrality = []

    for node in h.node_list:
        exp_degree_centrality.append(np.sum([node == i for i in edge_node_list]))

    degree_centrality = h.degree_centrality(weighted=False)

    for (act, exp) in zip(exp_degree_centrality, degree_centrality):
        assert act == exp

def test_edge_degree_centrality_unweighted():

    """
    Test the calculation of the degree centrality for a dual hypergraph
    with unit weights
    """

    data = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1]
    ])

    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )
    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)
    edge_node_list = [item for sublist in h.edge_list for item in sublist]

    exp_degree_centrality = []

    for edge in h.edge_list:
        exp_degree_centrality.append(len(edge))

    degree_centrality = h.degree_centrality(rep="dual", weighted=False)

    for (act, exp) in zip(exp_degree_centrality, degree_centrality):
        assert act == exp

def test_degree_centrality_exception_raised():
    """
    Tests that an exception is raised when an incorrect representation
    string is used
    """

    h = hgt.Hypergraph()
    with pytest.raises(Exception):
        h.degree_centrality(rep="oh no!")


def test_non_standard_weight_function():
    """
    Tests to make sure a user can specify a non standard weight function
    that is used in construct_hypergraph
    """
    @numba.jit(
        nopython=True,
        nogil=True,
        fastmath=True,
    )
    def unit_weights(data, inds):
        """
        This function returns a 1.0 for each edge, essentially creating an
        unweighted hypergraph.
        """

        return 1.0, 0.0, 0.0

    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd, weight_function=unit_weights)

    #assert len(h.edge_weights) == 1012
    assert (h.edge_weights == 1).all()


def test_non_standard_weight_function_with_optional_arguments():
    """
    Tests to make sure a user can specify a non standard weight function and
    specify optional arguments to be used in construct_hypergraph
    specify optional arguments to be used in construct_hypergraph
    """

    @numba.jit(
        nopython=True,
        nogil=True,
        fastmath=True,
    )
    def unit_weights(data, inds, *args):
        """
        This function returns a 1.0 divided by a number passed in as an optional
        argument.
        """
        print(args[0])
        return 1.0 / args[0], 0.0, 0.0

    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd, unit_weights, 2.0)

    #assert len(h.edge_weights) == 1012
    assert (h.edge_weights == 1/2).all()

def test_non_standard_weight_function_node_and_edge_weights_with_optional_arguments():
    """
    Tests to make sure a user can specify a non standard weight function and
    specify optional arguments to be used in construct_hypergraph
    specify optional arguments to be used in construct_hypergraph
    """

    @numba.jit(
        nopython=True,
        nogil=True,
        fastmath=True,
    )
    def unit_weights(data, inds, *args):
        """
        This function returns a 1.0 divided by a number passed in as an optional
        argument.
        """
        print(args[0])
        return 1.0 / args[0], 0.0, 0.0

    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd, unit_weights, 2.0)

    #assert len(h.edge_weights) == 1012
    assert (h.edge_weights == 1/2).all()
    assert (h.node_weights == 1/2).all()




def test_wilson_score_uncertainties():
    """
    Tests to make sure the edge weight variance is calculated correctly.
    """
    # Not sure I need these additional tests, since the existing tests now must
    # include the uncertainties.
    # non standard weight function with uncertainties
    # non standard weight function with optional args and uncertainties.
    def overlap_cooef_num_denom(data, edge):

        denominator = data.loc[:, edge].sum().min()
        numerator = (data.loc[:, edge].sum(axis=1) == len(edge)).sum()

        return (numerator, denominator)

    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    for i in range(len(h.edge_list)):
        vals = overlap_cooef_num_denom(data_pd, h.edge_list[i])
        wilson = smsp.proportion_confint(vals[0], vals[1], alpha=0.05, method="wilson")
        print(wilson, h.edge_weights_ci[i])
        assert (wilson - h.edge_weights_ci[i] < 0.001).all()

# test_bootstrap_standard_rep
# test_bootstrap_dual_rep
# test_bootstrap_bipartite_rep
# test_bootstrap_standard_rep_weighted_resultant
# test_bootstrap_dual_rep_weighted_resultant
# test_bootstrap_bipartite_rep_weighted_resultant

def test_bootstrap_standard_rep():
    """
    Test that the bootstrapped eigenvector centrality provides an answer that is
    close to the non bootstrapped answer (close means within 2 etimated standard
    deviations. Standard rep
    """
    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph(verbose=False)
    h.compute_hypergraph(data_pd)

    e_vec, e_vec_err = h.eigenvector_centrality()
    e_vec_boot, e_vec_err_boot = h.eigenvector_centrality(bootstrap_samples=10)

    np.testing.assert_array_less(np.abs(e_vec - e_vec_boot), 2 * np.sqrt(e_vec_err_boot))


def test_bootstrap_dual_rep():
    """
    Test that the bootstrapped eigenvector centrality provides an answer that is
    close to the non bootstrapped answer (close means within 2 etimated standard
    deviations. Dual rep
    """
    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph(verbose=False)
    h.compute_hypergraph(data_pd)

    e_vec, e_vec_err = h.eigenvector_centrality(rep="dual")
    e_vec_boot, e_vec_err_boot = h.eigenvector_centrality(rep="dual", bootstrap_samples=10)

    np.testing.assert_array_less(np.abs(e_vec - e_vec_boot), 2 * np.sqrt(e_vec_err_boot))


def test_bootstrap_bipartite_rep():
    """
    Test that the bootstrapped eigenvector centrality provides an answer that is
    close to the non bootstrapped answer (close means within 2 etimated standard
    deviations. Bipartite rep
    """
    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph(verbose=False)
    h.compute_hypergraph(data_pd)

    e_vec, e_vec_err = h.eigenvector_centrality(rep="bipartite")
    e_vec_boot, e_vec_err_boot = h.eigenvector_centrality(rep="bipartite", bootstrap_samples=10)

    tests = np.abs(e_vec - e_vec_boot) < 2 * np.sqrt(e_vec_err_boot)

    # This is a bit hacky...
    # Since bootstrapping is statistical, we would expect the difference to be
    # greater than twice the standard deviation about 5% of the time (if the
    # eigenvector elements are normally distributed)
    assert len(e_vec) - np.sum(tests) < 0.05 * len(e_vec)


def test_bootstrap_standard_rep_weighted_resultant():
    """
    Test that the bootstrapped eigenvector centrality provides an answer that is
    close to the non bootstrapped answer (close means within 2 etimated standard
    deviations. Standard rep, weighted resultant.
    """
    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph(verbose=False)
    h.compute_hypergraph(data_pd)

    e_vec, e_vec_err = h.eigenvector_centrality(weighted_resultant=True)
    e_vec_boot, e_vec_err_boot = h.eigenvector_centrality(
        weighted_resultant=True,
        bootstrap_samples=10
    )

    np.testing.assert_array_less(np.abs(e_vec - e_vec_boot), 2 * np.sqrt(e_vec_err_boot))


def test_bootstrap_dual_rep_weighted_resultant():
    """
    Test that the bootstrapped eigenvector centrality provides an answer that is
    close to the non bootstrapped answer (close means within 2 etimated standard
    deviations. Dual rep, weighted resultant.
    """
    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph(verbose=False)
    h.compute_hypergraph(data_pd)

    e_vec, e_vec_err = h.eigenvector_centrality(rep="dual", weighted_resultant=True)
    e_vec_boot, e_vec_err_boot = h.eigenvector_centrality(
        rep="dual",
        weighted_resultant=True,
        bootstrap_samples=10
    )

    tests = np.abs(e_vec - e_vec_boot) < 2 * np.sqrt(e_vec_err_boot)
    assert len(e_vec) - np.sum(tests) < 0.05 * len(e_vec)


def test_binomial_random_numbers():

    N = 50000
    p = 0.001
    samples = 1000
    iterations = 500
    p_values = np.zeros(iterations)

    for i in range(iterations):
        calc = hgt.randomize_weights(np.ones(samples) * N, np.ones(samples) * p)
        expected = sst.binom(N, p).rvs(samples) / N
        res = sst.ks_2samp(expected, calc)
        p_values[i] = res[1]


    # NOTE. This is checking <iterations> distributions at the 5% significance level.
    # i.e. at most 5% of <iterations> tests will have a p-value less than 0.05
    assert np.sum(p_values < 0.05) / iterations < 0.05


def test_benchmarking_compute_hypergraph(benchmark):

    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    benchmark(
        h.compute_hypergraph,
        data_pd
    )


def test_benchmarking_eigenvector_centrality(benchmark):

    n_people = 5000
    n_diseases = 10

    data = (np.random.rand(n_people, n_diseases) > 0.8).astype(np.uint8)
    data_pd = pd.DataFrame(
        data
    ).rename(
        columns={i: "disease_{}".format(i) for i in range(data.shape[1])}
    )

    h = hgt.Hypergraph()
    h.compute_hypergraph(data_pd)

    benchmark(
        h.eigenvector_centrality
    )