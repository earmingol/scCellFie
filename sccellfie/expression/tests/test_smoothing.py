import numpy as np
import pytest

from scipy.sparse import csr_matrix
from sccellfie.expression.smoothing import get_smoothing_matrix, smooth_expression_knn
from sccellfie.datasets.toy_inputs import create_controlled_adata

def test_get_smoothing_matrix():
    # Create a controlled adata object with known connectivities
    adata = create_controlled_adata()
    adata.uns['neighbors'] = {'connectivities_key': 'connectivities',
                              'distances_key': 'distances',
                              'params': {'method': 'manual',
                                         'metric': 'euclidean',
                                         'n_neighbors': 2,
                                         }}
    adata.obsp['distances'] = csr_matrix([[0., 3.464, 0., 0.],
                                          [3.464, 0., 0., 0.],
                                          [0., 0., 0., 4.899],
                                          [0., 0., 4.899, 0.]])
    adata.obsp['connectivities'] = csr_matrix([[0., 1., 0., 0.585],
                                               [1., 0., 0.585, 0.828],
                                               [0., 0.585, 0., 1.],
                                               [0.585, 0.828, 1., 0.]])

    # Test adjacency mode
    S_adjacency = get_smoothing_matrix(adata, mode='adjacency')
    expected_S_adjacency = np.array([[0, 1, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 0, 0, 1],
                                     [0, 0, 1, 0]])
    np.testing.assert_allclose(S_adjacency.toarray(), expected_S_adjacency)

    # Test connectivity mode
    S_connectivity = get_smoothing_matrix(adata, mode='connectivity')
    expected_S_connectivity = np.array([[0., 0.63091483, 0., 0.36908517],
                                        [0.41442188, 0., 0.2424368, 0.34314132],
                                        [0., 0.36908517, 0., 0.63091483],
                                        [0.2424368, 0.34314132, 0.41442188, 0.]])
    np.testing.assert_allclose(S_connectivity.toarray(), expected_S_connectivity)

    # Test invalid mode
    with pytest.raises(ValueError):
        get_smoothing_matrix(adata, mode='invalid')


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_smooth_expression_knn_alpha(alpha):
    # Create a controlled adata object with known expression values
    adata = create_controlled_adata()
    adata.uns['neighbors'] = {'connectivities_key': 'connectivities',
                              'distances_key': 'distances',
                              'params': {'method': 'manual',
                                         'metric': 'euclidean',
                                         'n_neighbors': 2,
                                         }}
    adata.obsp['distances'] = csr_matrix([[0., 3.464, 0., 0.],
                                          [3.464, 0., 0., 0.],
                                          [0., 0., 0., 4.899],
                                          [0., 0., 4.899, 0.]])
    adata.obsp['connectivities'] = csr_matrix([[0., 1., 0., 0.585],
                                               [1., 0., 0.585, 0.828],
                                               [0., 0.585, 0., 1.],
                                               [0.585, 0.828, 1., 0.]])

    # Test smoothing with varying alpha values
    smooth_expression_knn(adata, key_added=f'smoothed_X_alpha_{alpha}', alpha=alpha)

    # Expected smoothed expression values for each alpha
    expected_smoothed_X = {
        0.0: np.array([[1,  2,  0],
                       [3, 4, 2],
                       [5, 6, 10],
                       [7, 8, 6,]]),
        0.1: np.array([[1.34763407, 2.34763407, 0.34763407],
                       [3.10285951, 4.10285951, 2.24832159],
                       [5.05236593, 6.05236593, 9.45236593],
                       [6.63439702, 7.63439702, 5.88305015]]),
        0.5: np.array([[2.73817035, 3.73817035, 1.73817035],
                       [3.51429755, 4.51429755, 3.24160796],
                       [5.26182965, 6.26182965, 7.26182965],
                       [5.17198508, 6.17198508, 5.41525073]]),
        0.9: np.array([[4.12870662, 5.12870662, 3.12870662],
                       [3.9257356 , 4.9257356 , 4.23489432],
                       [5.47129338, 6.47129338, 5.07129338],
                       [3.70957315, 4.70957315, 4.94745131]]),
        1.0: np.array([[4.47634069, 5.47634069, 3.47634069],
                       [4.02859511, 5.02859511, 4.48321591],
                       [5.52365931, 6.52365931, 4.52365931],
                       [3.34397016, 4.34397016, 4.83050145]])
    }

    # Check the smoothed expression values for each alpha
    np.testing.assert_allclose(adata.layers[f'smoothed_X_alpha_{alpha}'].toarray(), expected_smoothed_X[alpha])


@pytest.mark.parametrize("use_raw", [False, True])
def test_smooth_expression_knn_raw(use_raw):
    # Create a controlled adata object with known expression values
    adata = create_controlled_adata()
    adata.uns['neighbors'] = {'connectivities_key': 'connectivities',
                              'distances_key': 'distances',
                              'params': {'method': 'manual',
                                         'metric': 'euclidean',
                                         'n_neighbors': 2,
                                         }}
    adata.obsp['distances'] = csr_matrix([[0., 3.464, 0., 0.],
                                          [3.464, 0., 0., 0.],
                                          [0., 0., 0., 4.899],
                                          [0., 0., 4.899, 0.]])
    adata.obsp['connectivities'] = csr_matrix([[0., 1., 0., 0.585],
                                               [1., 0., 0.585, 0.828],
                                               [0., 0.585, 0., 1.],
                                               [0.585, 0.828, 1., 0.]])

    # Test smoothing with varying alpha values
    alpha = 0.5
    smooth_expression_knn(adata, key_added=f'smoothed_X_alpha_{alpha}', alpha=alpha, use_raw=use_raw)

    # Expected smoothed expression values for each alpha
    expected_smoothed_X = np.array([[2.73817035, 3.73817035, 1.73817035],
                                    [3.51429755, 4.51429755, 3.24160796],
                                    [5.26182965, 6.26182965, 7.26182965],
                                    [5.17198508, 6.17198508, 5.41525073]
                                    ])

    # Check the smoothed expression values for each alpha
    np.testing.assert_allclose(adata.layers[f'smoothed_X_alpha_{alpha}'].toarray(), expected_smoothed_X)