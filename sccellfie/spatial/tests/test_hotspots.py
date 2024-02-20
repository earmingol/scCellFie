import pytest
import pandas as pd

from sccellfie.spatial.hotspots import obtain_hotspots, summarize_hotspots, compute_hotspots
from sccellfie.tests.toy_inputs import create_random_adata_with_spatial

@pytest.mark.parametrize("use_raw", [False, True])
def test_obtain_hotspots(use_raw):
    adata =  create_random_adata_with_spatial(layers='random_layer') # Layer just for coverage purposes
    hotspots = obtain_hotspots(adata, use_raw=use_raw)
    assert isinstance(hotspots, dict)
    assert len(hotspots) == adata.shape[1]


def test_summarize_hotspots():
    adata = create_random_adata_with_spatial()
    hotspots = obtain_hotspots(adata)
    hotspot_df = summarize_hotspots(hotspots)
    assert isinstance(hotspot_df, pd.DataFrame)
    assert len(hotspot_df) == adata.shape[1]
    assert all(col in hotspot_df.columns for col in ['Var-Name', 'Mean-Hotspot-Z', 'Median-Hotspot-Z',
                                                      'Hotspot-Proportion', 'Coldspot-Proportion',
                                                      'Significant-Proportion'])


def test_compute_hotspots():
    adata = create_random_adata_with_spatial()
    compute_hotspots(adata)
    assert 'hotspots' in adata.uns
    assert 'hotspot_df' in adata.uns['hotspots']
    assert isinstance(adata.uns['hotspots']['hotspots'], dict)
    assert isinstance(adata.uns['hotspots']['hotspot_df'], pd.DataFrame)

