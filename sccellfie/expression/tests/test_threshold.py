import pytest
import pandas as pd

from pandas.testing import assert_frame_equal
from sccellfie.expression.thresholds import get_local_mean_threshold, get_global_mean_threshold, get_global_percentile_threshold, get_local_percentile_threshold
from sccellfie.tests.toy_inputs import create_controlled_adata


# data = np.array([
#         [1, 2, 0],  # Cell1
#         [3, 4, 2],  # Cell2
#         [5, 6, 10],  # Cell3
#         [7, 8, 6],  # Cell4
#     ])


@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-0.75': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-0.75': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_local_percentile_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    percentile = 0.75
    thresholds = get_local_percentile_threshold(adata,
                                                percentile=percentile,
                                                use_raw=use_raw,
                                                lower_bound=lower_bound,
                                                upper_bound=upper_bound,
                                                exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({f'threshold-{percentile}': [6, 7, 8]},
                                  index=['gene1', 'gene2', 'gene3'],
                                  dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = [f'threshold-{percentile}']
            expected_values[expected_values < lb] = lb[expected_values < lb]
        else:
            expected_values[expected_values < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]
        else:
            expected_values[expected_values > upper_bound] = upper_bound

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-{}'.format(percentile) in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert thresholds.equals(expected_values), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert all(thresholds >= lower_bound), "Lower bound not respected"
    if upper_bound is not None:
        assert all(thresholds <= upper_bound), "Upper bound not respected"


@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-0.75': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-0.75': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_global_percentile_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    percentile = 0.75
    thresholds = get_global_percentile_threshold(adata,
                                                 percentile=percentile,
                                                 use_raw=use_raw,
                                                 lower_bound=lower_bound,
                                                 upper_bound=upper_bound,
                                                 exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({f'threshold-{percentile}': [6.5, 6.5, 6.5]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = [f'threshold-{percentile}']
            expected_values[expected_values < lb] = lb[expected_values < lb]
        else:
            expected_values[expected_values < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]
        else:
            expected_values[expected_values > upper_bound] = upper_bound

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-{}'.format(percentile) in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert thresholds.equals(expected_values), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert all(thresholds >= lower_bound), "Lower bound not respected"
    if upper_bound is not None:
        assert all(thresholds <= upper_bound), "Upper bound not respected"


@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-mean': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-mean': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_local_mean_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    thresholds = get_local_mean_threshold(adata,
                                          use_raw=use_raw,
                                          lower_bound=lower_bound,
                                          upper_bound=upper_bound,
                                          exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({'threshold-mean': [4. , 5. , 4.5]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    if exclude_zeros:
        expected_values.loc['gene3', 'threshold-mean'] = 6.0

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            expected_values[expected_values < lb] = lb[expected_values < lb]
        else:
            expected_values[expected_values < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]
        else:
            expected_values[expected_values > upper_bound] = upper_bound

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-mean' in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert thresholds.equals(expected_values), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert all(thresholds >= lower_bound), "Lower bound not respected"
    if upper_bound is not None:
        assert all(thresholds <= upper_bound), "Upper bound not respected"

@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-mean': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-mean': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_global_mean_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    thresholds = get_global_mean_threshold(adata,
                                           use_raw=use_raw,
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({'threshold-mean': [4.5 , 4.5 , 4.5]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    if exclude_zeros:
        expected_values = pd.DataFrame({'threshold-mean': [4.909091, 4.909091, 4.909091]},
                                       index=['gene1', 'gene2', 'gene3'],
                                       dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            expected_values[expected_values < lb] = lb[expected_values < lb]
        else:
            expected_values[expected_values < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]
        else:
            expected_values[expected_values > upper_bound] = upper_bound

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-mean' in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    print(thresholds, expected_values)
    assert_frame_equal(thresholds, expected_values, check_exact=False), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert all(thresholds >= lower_bound), "Lower bound not respected"
    if upper_bound is not None:
        assert all(thresholds <= upper_bound), "Upper bound not respected"