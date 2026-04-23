import pytest
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from sccellfie.datasets.toy_inputs import create_controlled_adata, create_controlled_adata_with_spatial
from sccellfie.preprocessing.complexes import add_complexes_to_adata, make_complex_name, prepare_var_pairs
from sccellfie.expression.aggregation import agg_expression_cells
from sccellfie.communication import compute_communication_scores, compute_local_colocalization_scores


@pytest.fixture
def adata_fixture():
    return create_controlled_adata()


@pytest.fixture
def adata_spatial_fixture():
    return create_controlled_adata_with_spatial()


# ---- Tests for add_complexes_to_adata ----

def test_add_complexes_basic(adata_fixture):
    """Test basic functionality: new var added, sparse preserved, correct values."""
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min')

    assert 'gene1&gene2' in adata_fixture.var_names
    assert adata_fixture.n_vars == 4
    assert issparse(adata_fixture.X)

    # min([1,2])=1, min([3,4])=3, min([5,6])=5, min([7,8])=7
    complex_expr = adata_fixture[:, 'gene1&gene2'].X.toarray().flatten()
    np.testing.assert_array_almost_equal(complex_expr, [1, 3, 5, 7])


def test_add_complexes_agg_methods(adata_fixture):
    """Test all aggregation methods produce correct values."""
    # min
    adata_min = adata_fixture.copy()
    add_complexes_to_adata(adata_min, {'c_min': ['gene1', 'gene2']}, agg_method='min')
    vals_min = adata_min[:, 'c_min'].X.toarray().flatten()
    np.testing.assert_array_almost_equal(vals_min, [1, 3, 5, 7])

    # mean
    adata_mean = adata_fixture.copy()
    add_complexes_to_adata(adata_mean, {'c_mean': ['gene1', 'gene2']}, agg_method='mean')
    vals_mean = adata_mean[:, 'c_mean'].X.toarray().flatten()
    np.testing.assert_array_almost_equal(vals_mean, [1.5, 3.5, 5.5, 7.5])

    # gmean: sqrt(1*2), sqrt(3*4), sqrt(5*6), sqrt(7*8)
    adata_gmean = adata_fixture.copy()
    add_complexes_to_adata(adata_gmean, {'c_gmean': ['gene1', 'gene2']}, agg_method='gmean')
    vals_gmean = adata_gmean[:, 'c_gmean'].X.toarray().flatten()
    expected_gmean = [np.sqrt(2), np.sqrt(12), np.sqrt(30), np.sqrt(56)]
    np.testing.assert_array_almost_equal(vals_gmean, expected_gmean)


def test_add_complexes_three_subunits(adata_fixture):
    """Test complex with three subunits."""
    complexes = {'all_genes': ['gene1', 'gene2', 'gene3']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min')

    # min([1,2,0])=0, min([3,4,2])=2, min([5,6,10])=5, min([7,8,6])=6
    complex_expr = adata_fixture[:, 'all_genes'].X.toarray().flatten()
    np.testing.assert_array_almost_equal(complex_expr, [0, 2, 5, 6])


def test_add_complexes_fraction_correctness(adata_fixture):
    """Test that fraction_above is computed on complex aggregate, not individual genes."""
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min')

    fractions = agg_expression_cells(
        adata_fixture, 'group',
        gene_symbols=['gene1&gene2'],
        agg_func='fraction_above',
        threshold=2
    )

    # Complex min values: cell1=1, cell2=3, cell3=5, cell4=7
    # Group A (cells 1,2): values [1,3], fraction above 2 = 0.5
    # Group B (cells 3,4): values [5,7], fraction above 2 = 1.0
    assert np.isclose(fractions.loc['A', 'gene1&gene2'], 0.5)
    assert np.isclose(fractions.loc['B', 'gene1&gene2'], 1.0)


def test_add_complexes_with_communication_scores(adata_fixture):
    """End-to-end: add complex, then run communication scoring."""
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min')

    var_pairs = [('gene3', 'gene1&gene2')]
    scores = compute_communication_scores(adata_fixture, 'group', var_pairs)

    assert isinstance(scores, pd.DataFrame)
    assert len(scores) == 4  # 2x2 cell type combinations

    # Sender A, Receiver B: gene3 mean in A = (0+2)/2 = 1, complex mean in B = (5+7)/2 = 6
    # gmean = sqrt(1 * 6) = sqrt(6)
    ab_score = scores[
        (scores['sender_celltype'] == 'A') & (scores['receiver_celltype'] == 'B')
    ]['score'].iloc[0]
    assert np.isclose(ab_score, np.sqrt(1 * 6))


def test_add_complexes_missing_subunit(adata_fixture):
    """Test error when subunit gene doesn't exist."""
    complexes = {'bad_complex': ['gene1', 'nonexistent_gene']}
    with pytest.raises(ValueError, match="not found in adata.var_names"):
        add_complexes_to_adata(adata_fixture, complexes)


def test_add_complexes_existing_name(adata_fixture):
    """Test error when complex name collides with existing var_name."""
    complexes = {'gene1': ['gene1', 'gene2']}
    with pytest.raises(ValueError, match="already exist in adata.var_names"):
        add_complexes_to_adata(adata_fixture, complexes)


def test_add_complexes_single_subunit(adata_fixture):
    """Test that single-subunit complex equals the gene expression."""
    complexes = {'gene1_solo': ['gene1']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min')

    complex_expr = adata_fixture[:, 'gene1_solo'].X.toarray().flatten()
    gene1_expr = adata_fixture[:, 'gene1'].X.toarray().flatten()
    np.testing.assert_array_almost_equal(complex_expr, gene1_expr)


def test_add_complexes_copy(adata_fixture):
    """Test copy=True returns modified copy, original unchanged."""
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    result = add_complexes_to_adata(adata_fixture, complexes, copy=True)

    assert adata_fixture.n_vars == 3  # Original unchanged
    assert result.n_vars == 4
    assert 'gene1&gene2' in result.var_names
    assert 'gene1&gene2' not in adata_fixture.var_names


def test_add_complexes_inplace(adata_fixture):
    """Test copy=False modifies in place and returns None."""
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    result = add_complexes_to_adata(adata_fixture, complexes, copy=False)

    assert result is None
    assert adata_fixture.n_vars == 4
    assert 'gene1&gene2' in adata_fixture.var_names


def test_add_complexes_var_metadata(adata_fixture):
    """Test that complex metadata is stored in adata.var."""
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min')

    assert adata_fixture.var.loc['gene1&gene2', 'is_complex'] == True
    assert adata_fixture.var.loc['gene1&gene2', 'complex_subunits'] == 'gene1|gene2'
    assert adata_fixture.var.loc['gene1&gene2', 'complex_agg_method'] == 'min'
    assert adata_fixture.var.loc['gene1', 'is_complex'] == False


def test_add_complexes_with_layer(adata_fixture):
    """Test complex computation from a specific layer."""
    adata_fixture.layers['test'] = adata_fixture.X * 2
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min', layer='test')

    # Complex in layer 'test' should be min(2*gene1, 2*gene2) per cell
    # = min([2,4])=2, min([6,8])=6, min([10,12])=10, min([14,16])=14
    layer_expr = adata_fixture.layers['test']
    if issparse(layer_expr):
        layer_expr = layer_expr.toarray()
    complex_idx = list(adata_fixture.var_names).index('gene1&gene2')
    complex_layer_vals = layer_expr[:, complex_idx]
    np.testing.assert_array_almost_equal(complex_layer_vals, [2, 6, 10, 14])


def test_add_complexes_multiple(adata_fixture):
    """Test adding multiple complexes in one call."""
    complexes = {
        'gene1&gene2': ['gene1', 'gene2'],
        'gene2&gene3': ['gene2', 'gene3'],
    }
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min')

    assert adata_fixture.n_vars == 5
    assert 'gene1&gene2' in adata_fixture.var_names
    assert 'gene2&gene3' in adata_fixture.var_names

    # gene1&gene2 min: [1,3,5,7]
    expr1 = adata_fixture[:, 'gene1&gene2'].X.toarray().flatten()
    np.testing.assert_array_almost_equal(expr1, [1, 3, 5, 7])

    # gene2&gene3 min: min([2,0])=0, min([4,2])=2, min([6,10])=6, min([8,6])=6
    expr2 = adata_fixture[:, 'gene2&gene3'].X.toarray().flatten()
    np.testing.assert_array_almost_equal(expr2, [0, 2, 6, 6])


def test_add_complexes_invalid_agg_method(adata_fixture):
    """Test error for unknown aggregation method."""
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    with pytest.raises(ValueError, match="Invalid agg_method"):
        add_complexes_to_adata(adata_fixture, complexes, agg_method='invalid')


def test_add_complexes_gmean_with_zeros(adata_fixture):
    """Test gmean handles zero expression gracefully (no NaN)."""
    # gene3 has value 0 for cell1
    complexes = {'gene1&gene3': ['gene1', 'gene3']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='gmean')

    complex_expr = adata_fixture[:, 'gene1&gene3'].X.toarray().flatten()
    assert not np.any(np.isnan(complex_expr))
    # Cell1: gmean(1, 0) ≈ 0 due to clipping
    assert complex_expr[0] < 0.01


def test_add_complexes_dense_matrix(adata_fixture):
    """Test that function works with dense X matrix."""
    adata_fixture.X = adata_fixture.X.toarray()
    assert not issparse(adata_fixture.X)

    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    add_complexes_to_adata(adata_fixture, complexes, agg_method='min')

    assert not issparse(adata_fixture.X)
    assert adata_fixture.n_vars == 4
    complex_expr = adata_fixture[:, 'gene1&gene2'].X.flatten()
    np.testing.assert_array_almost_equal(complex_expr, [1, 3, 5, 7])


def test_add_complexes_with_colocalization(adata_spatial_fixture):
    """End-to-end: add complex, then run colocalization scoring."""
    complexes = {'gene1&gene2': ['gene1', 'gene2']}
    add_complexes_to_adata(adata_spatial_fixture, complexes, agg_method='min')

    scores = compute_local_colocalization_scores(
        adata_spatial_fixture,
        var1='gene3',
        var2='gene1&gene2',
        neighbors_radius=3.0,
        method='concordance',
        min_neighbors=2,
        inplace=False,
    )

    assert scores is not None
    assert len(scores) == adata_spatial_fixture.n_obs
    assert scores.dtype == np.float64


# ---- Tests for make_complex_name ----

def test_make_complex_name():
    """Test canonical name generation with sorting and separator."""
    assert make_complex_name(['ITGB1', 'ITGA4']) == 'ITGA4&ITGB1'
    assert make_complex_name(['ITGA4', 'ITGB1']) == 'ITGA4&ITGB1'
    assert make_complex_name(['ITGA4', 'ITGB1'], separator='_') == 'ITGA4_ITGB1'
    assert make_complex_name(['A', 'C', 'B']) == 'A&B&C'
    assert make_complex_name(['SINGLE']) == 'SINGLE'


# ---- Tests for prepare_var_pairs ----

def test_prepare_var_pairs_basic(adata_fixture):
    """Test that list elements are detected, added to adata, and pairs normalized."""
    var_pairs = [(['gene1', 'gene2'], 'gene3')]
    normalized = prepare_var_pairs(adata_fixture, var_pairs)

    assert normalized == [('gene1&gene2', 'gene3')]
    assert 'gene1&gene2' in adata_fixture.var_names
    assert adata_fixture.n_vars == 4


def test_prepare_var_pairs_both_complex(adata_fixture):
    """Test both ligand and receptor as complexes."""
    var_pairs = [(['gene1', 'gene2'], ['gene2', 'gene3'])]
    normalized = prepare_var_pairs(adata_fixture, var_pairs)

    assert normalized == [('gene1&gene2', 'gene2&gene3')]
    assert 'gene1&gene2' in adata_fixture.var_names
    assert 'gene2&gene3' in adata_fixture.var_names
    assert adata_fixture.n_vars == 5


def test_prepare_var_pairs_mixed(adata_fixture):
    """Test mix of single-string and list elements across pairs."""
    var_pairs = [
        (['gene1', 'gene2'], 'gene3'),
        ('gene1', ['gene2', 'gene3']),
        ('gene1', 'gene2'),
    ]
    normalized = prepare_var_pairs(adata_fixture, var_pairs)

    assert normalized == [
        ('gene1&gene2', 'gene3'),
        ('gene1', 'gene2&gene3'),
        ('gene1', 'gene2'),
    ]
    assert adata_fixture.n_vars == 5  # 3 original + 2 complexes


def test_prepare_var_pairs_no_complexes(adata_fixture):
    """Test that string-only pairs pass through unchanged."""
    var_pairs = [('gene1', 'gene2'), ('gene2', 'gene3')]
    normalized = prepare_var_pairs(adata_fixture, var_pairs)

    assert normalized == var_pairs
    assert adata_fixture.n_vars == 3  # No new vars added


def test_prepare_var_pairs_duplicate_complex(adata_fixture):
    """Test that the same complex appearing in multiple pairs is only added once."""
    var_pairs = [
        (['gene1', 'gene2'], 'gene3'),
        (['gene1', 'gene2'], 'gene1'),
    ]
    normalized = prepare_var_pairs(adata_fixture, var_pairs)

    assert normalized == [('gene1&gene2', 'gene3'), ('gene1&gene2', 'gene1')]
    assert adata_fixture.n_vars == 4  # Only one complex added


def test_prepare_var_pairs_already_in_adata(adata_fixture):
    """Test that complexes already in adata are skipped."""
    add_complexes_to_adata(adata_fixture, {'gene1&gene2': ['gene1', 'gene2']})
    assert adata_fixture.n_vars == 4

    var_pairs = [(['gene1', 'gene2'], 'gene3')]
    normalized = prepare_var_pairs(adata_fixture, var_pairs)

    assert normalized == [('gene1&gene2', 'gene3')]
    assert adata_fixture.n_vars == 4  # Not added again


def test_prepare_var_pairs_custom_separator(adata_fixture):
    """Test custom separator for complex naming."""
    var_pairs = [(['gene1', 'gene2'], 'gene3')]
    normalized = prepare_var_pairs(adata_fixture, var_pairs, complex_sep='_')

    assert normalized == [('gene1_gene2', 'gene3')]
    assert 'gene1_gene2' in adata_fixture.var_names


def test_prepare_var_pairs_end_to_end(adata_fixture):
    """End-to-end: prepare_var_pairs then compute_communication_scores."""
    var_pairs = [('gene3', ['gene1', 'gene2'])]
    normalized = prepare_var_pairs(adata_fixture, var_pairs, agg_method='min')

    scores = compute_communication_scores(adata_fixture, 'group', normalized)

    assert isinstance(scores, pd.DataFrame)
    assert len(scores) == 4

    # Sender A, Receiver B: gene3 mean in A = 1, complex min mean in B = 6
    ab_score = scores[
        (scores['sender_celltype'] == 'A') & (scores['receiver_celltype'] == 'B')
    ]['score'].iloc[0]
    assert np.isclose(ab_score, np.sqrt(1 * 6))


def test_prepare_var_pairs_sorts_subunits(adata_fixture):
    """Test that subunit order doesn't matter — same complex name generated."""
    var_pairs = [(['gene2', 'gene1'], 'gene3')]
    normalized = prepare_var_pairs(adata_fixture, var_pairs)

    # Sorted: gene1&gene2 regardless of input order
    assert normalized == [('gene1&gene2', 'gene3')]
