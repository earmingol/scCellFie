import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from sccellfie.reports.completeness import (
    compute_cell_completeness,
    compute_dataset_completeness,
    generate_completeness_report,
)
from sccellfie.stats.ablation import compute_gene_ablation_impact


# ---------------- fixtures ----------------


@pytest.fixture
def simple_gpr():
    """
    R1: A and B   (both essential at fz=1.0)
    R2: C or D    (none essential)
    R3: E         (E essential)
    """
    return {"R1": "A and B", "R2": "C or D", "R3": "E"}


@pytest.fixture
def simple_task_by_rxn():
    # T1 -> R1, T2 -> R2, T3 -> R3 (one reaction per task).
    return pd.DataFrame(
        np.eye(3, dtype=int),
        index=["T1", "T2", "T3"],
        columns=["R1", "R2", "R3"],
    )


def _make_adata(var_names, X=None):
    n_cells = 3
    if X is None:
        X = np.ones((n_cells, len(var_names)), dtype=float)
    adata = sc.AnnData(X=np.asarray(X, dtype=float))
    adata.var_names = list(var_names)
    adata.obs_names = [f"cell{i}" for i in range(1, n_cells + 1)]
    return adata


# ---------------- dataset-level ----------------


def test_dataset_task_completeness_missing_essential(simple_gpr, simple_task_by_rxn):
    # adata has A, C, D, E but NOT B -> T1 loses an essential gene.
    adata = _make_adata(["A", "C", "D", "E"])

    out = compute_dataset_completeness(adata, simple_gpr, simple_task_by_rxn)
    task_df = out["task_completeness"]

    # T1 essentials are {A, B}; B missing.
    assert task_df.loc["T1", "n_genes_essential"] == 2
    assert task_df.loc["T1", "n_missing_essential"] == 1
    assert task_df.loc["T1", "fraction_present_essential"] == pytest.approx(0.5)
    assert task_df.loc["T1", "impact_weighted_completeness_essential"] == pytest.approx(0.0)
    assert task_df.loc["T1", "missing_genes_essential"] == "B"

    # All-scope for T1 equals essential here (R1 = "A and B").
    assert task_df.loc["T1", "n_genes_all"] == 2
    assert task_df.loc["T1", "n_missing_all"] == 1
    assert task_df.loc["T1", "fraction_present_all"] == pytest.approx(0.5)
    assert task_df.loc["T1", "impact_weighted_completeness_all"] == pytest.approx(0.0)

    # T2 has no essentials (OR-only) and all genes (C, D) present.
    assert task_df.loc["T2", "n_genes_essential"] == 0
    assert task_df.loc["T2", "fraction_present_essential"] == pytest.approx(1.0)
    assert task_df.loc["T2", "impact_weighted_completeness_essential"] == pytest.approx(1.0)
    assert task_df.loc["T2", "n_genes_all"] == 2
    assert task_df.loc["T2", "n_missing_all"] == 0

    # T3 essential {E}, present.
    assert task_df.loc["T3", "fraction_present_essential"] == pytest.approx(1.0)
    assert task_df.loc["T3", "impact_weighted_completeness_essential"] == pytest.approx(1.0)


def test_dataset_reaction_completeness(simple_gpr, simple_task_by_rxn):
    adata = _make_adata(["A", "C", "D", "E"])

    out = compute_dataset_completeness(adata, simple_gpr, simple_task_by_rxn)
    rxn_df = out["reaction_completeness"]

    # R1: A and B -> essentials {A, B}, B missing.
    assert rxn_df.loc["R1", "n_genes_essential"] == 2
    assert rxn_df.loc["R1", "n_missing_essential"] == 1
    assert rxn_df.loc["R1", "fraction_present_essential"] == pytest.approx(0.5)
    assert rxn_df.loc["R1", "impact_weighted_completeness_essential"] == pytest.approx(0.0)

    # R2: C or D -> essentials {}, all {C, D}, both present.
    assert rxn_df.loc["R2", "n_genes_essential"] == 0
    assert rxn_df.loc["R2", "fraction_present_essential"] == pytest.approx(1.0)
    assert rxn_df.loc["R2", "n_genes_all"] == 2
    assert rxn_df.loc["R2", "n_missing_all"] == 0

    # R3: E -> essential {E}, present.
    assert rxn_df.loc["R3", "n_genes_essential"] == 1
    assert rxn_df.loc["R3", "n_missing_essential"] == 0
    assert rxn_df.loc["R3", "impact_weighted_completeness_essential"] == pytest.approx(1.0)


def test_dataset_overall_summary(simple_gpr, simple_task_by_rxn):
    adata = _make_adata(["A", "C", "D", "E"])
    out = compute_dataset_completeness(adata, simple_gpr, simple_task_by_rxn)
    summary = out["overall_summary"].iloc[0]

    assert summary["n_tasks_total"] == 3
    # T1 compromised (both essential and all), T2 and T3 fine.
    assert summary["n_tasks_compromised_essential"] == 1
    assert summary["n_tasks_compromised_all"] == 1
    assert summary["n_tasks_fully_covered_essential"] == 2  # T2 (0 essentials) and T3.
    assert summary["n_tasks_fully_covered_all"] == 2       # T2 and T3 (both fully present).
    # DB has 5 genes (A-E); adata has 4 (B missing).
    assert summary["n_genes_in_db_total"] == 5
    assert summary["n_genes_in_db_present_in_adata"] == 4
    assert summary["fraction_db_genes_present"] == pytest.approx(0.8)


def test_dataset_all_genes_present(simple_gpr, simple_task_by_rxn):
    adata = _make_adata(["A", "B", "C", "D", "E"])
    out = compute_dataset_completeness(adata, simple_gpr, simple_task_by_rxn)
    task_df = out["task_completeness"]
    assert (task_df["fraction_present_essential"] == 1.0).all()
    assert (task_df["impact_weighted_completeness_essential"] == 1.0).all()
    assert (task_df["fraction_present_all"] == 1.0).all()
    assert (task_df["impact_weighted_completeness_all"] == 1.0).all()


def test_dataset_threshold_affects_essentials(simple_gpr, simple_task_by_rxn):
    """Lower threshold on rel_change pulls more genes into the essential set."""
    # Task T with R1="A and B" and R2="C" (genes A, B, C). Each zeros half the task.
    gpr = {"R1": "A and B", "R2": "C"}
    tbr = pd.DataFrame([[1, 1]], index=["T"], columns=["R1", "R2"])
    adata = _make_adata(["A", "C"])  # B missing

    # With fraction_zeroed @ 1.0, essentials = {} (no gene zeroes the MT score).
    out_fz = compute_dataset_completeness(adata, gpr, tbr, metric="fraction_zeroed", threshold=1.0)
    row = out_fz["task_completeness"].loc["T"]
    assert row["n_genes_essential"] == 0
    # All-scope: B missing => fraction_present_all = 2/3.
    assert row["fraction_present_all"] == pytest.approx(2 / 3)

    # With rel_change @ 0.5, essentials = {A, B, C} (each reduces MTS by 0.5).
    out_rc = compute_dataset_completeness(adata, gpr, tbr, metric="rel_change", threshold=0.5)
    row_rc = out_rc["task_completeness"].loc["T"]
    assert row_rc["n_genes_essential"] == 3
    assert row_rc["n_missing_essential"] == 1


# ---------------- per-cell ----------------


def test_cell_completeness_basic_patterns(simple_gpr, simple_task_by_rxn):
    # All genes present in adata; cell-level zeros vary.
    # cell1: all > 0 (complete).
    # cell2: A == 0 (hits T1).
    # cell3: all zero (hits everything).
    X = np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    adata = _make_adata(["A", "B", "C", "D", "E"], X=X)

    out = compute_cell_completeness(adata, simple_gpr, simple_task_by_rxn, return_matrix=True)
    per_cell = out["per_cell"]
    mat_ess = out["matrix_essential"]
    mat_all = out["matrix_all"]

    # cell1: no missing anywhere -> 1.0 everywhere.
    assert per_cell.loc["cell1", "completeness_essential"] == pytest.approx(1.0)
    assert per_cell.loc["cell1", "completeness_all"] == pytest.approx(1.0)

    # cell2: A=0. T1 essential impact=1.0 -> T1 completeness 0. T2 unaffected. T3 unaffected.
    assert mat_ess.loc["cell2", "T1"] == pytest.approx(0.0)
    assert mat_ess.loc["cell2", "T2"] == pytest.approx(1.0)
    assert mat_ess.loc["cell2", "T3"] == pytest.approx(1.0)
    # Mean = 2/3.
    assert per_cell.loc["cell2", "completeness_essential"] == pytest.approx(2 / 3)

    # cell3: all zero. T1 => 0 (essentials A,B both missing); T2 => 0 under "all" scope
    # (both C and D zero -> impact sum = 0 from rel_change, since OR has rel=0 each ->
    # actually the impact-weighted for T2 with both C,D zero is 0, so completeness stays 1)
    # Verify: rel_change for C and D on T2 are both 0 (OR), so total impact = 0 -> 1.
    assert mat_ess.loc["cell3", "T1"] == pytest.approx(0.0)
    assert mat_ess.loc["cell3", "T3"] == pytest.approx(0.0)
    # T2 with zero impact contributions stays at 1 under both scopes.
    assert mat_ess.loc["cell3", "T2"] == pytest.approx(1.0)
    assert mat_all.loc["cell3", "T2"] == pytest.approx(1.0)


def test_cell_completeness_writes_obs(simple_gpr, simple_task_by_rxn):
    X = np.ones((3, 5), dtype=float)
    adata = _make_adata(["A", "B", "C", "D", "E"], X=X)

    compute_cell_completeness(
        adata, simple_gpr, simple_task_by_rxn, write_to_obs=True
    )
    assert "completeness_essential" in adata.obs.columns
    assert "completeness_all" in adata.obs.columns
    np.testing.assert_allclose(adata.obs["completeness_essential"].values, 1.0)
    np.testing.assert_allclose(adata.obs["completeness_all"].values, 1.0)


def test_cell_completeness_dataset_absent_gene_is_constant(simple_gpr, simple_task_by_rxn):
    # B is missing from adata -> every cell inherits the dataset-absent impact for T1.
    X = np.ones((3, 4), dtype=float)
    adata = _make_adata(["A", "C", "D", "E"], X=X)

    out = compute_cell_completeness(
        adata, simple_gpr, simple_task_by_rxn, return_matrix=True, write_to_obs=False
    )
    mat_ess = out["matrix_essential"]
    # T1 essential impact for missing B = 1.0 -> completeness 0 for every cell.
    assert mat_ess["T1"].eq(0.0).all()
    # T3 completely present.
    assert mat_ess["T3"].eq(1.0).all()


def test_cell_completeness_return_matrix_false(simple_gpr, simple_task_by_rxn):
    X = np.ones((3, 5), dtype=float)
    adata = _make_adata(["A", "B", "C", "D", "E"], X=X)

    out = compute_cell_completeness(
        adata, simple_gpr, simple_task_by_rxn, return_matrix=False
    )
    assert out["matrix_essential"] is None
    assert out["matrix_all"] is None
    assert isinstance(out["per_cell"], pd.DataFrame)
    assert set(out["per_cell"].columns) == {"completeness_essential", "completeness_all"}


def test_cell_completeness_sparse_input(simple_gpr, simple_task_by_rxn):
    X = sparse.csr_matrix(np.ones((3, 5), dtype=float))
    adata = sc.AnnData(X=X)
    adata.var_names = ["A", "B", "C", "D", "E"]
    adata.obs_names = ["cell1", "cell2", "cell3"]
    out = compute_cell_completeness(
        adata, simple_gpr, simple_task_by_rxn, write_to_obs=False
    )
    np.testing.assert_allclose(out["per_cell"]["completeness_essential"].values, 1.0)


# ---------------- pass-through of precomputed impacts ----------------


def test_dataset_completeness_precomputed_impact(simple_gpr, simple_task_by_rxn):
    adata = _make_adata(["A", "C", "D", "E"])
    task_imp = compute_gene_ablation_impact(simple_gpr, simple_task_by_rxn, disable_pbar=True)

    # Identity task-by-rxn for reaction impact.
    rxn_ids = list(simple_gpr.keys())
    identity_tbr = pd.DataFrame(
        np.eye(len(rxn_ids), dtype=int), index=rxn_ids, columns=rxn_ids
    )
    rxn_imp = compute_gene_ablation_impact(simple_gpr, identity_tbr, disable_pbar=True)

    out = compute_dataset_completeness(
        adata,
        simple_gpr,
        simple_task_by_rxn,
        ablation_impact=task_imp,
        reaction_impact=rxn_imp,
    )
    assert out["task_completeness"].loc["T1", "fraction_present_essential"] == pytest.approx(0.5)


# ---------------- convenience wrapper ----------------


def test_generate_completeness_report_runs_both(simple_gpr, simple_task_by_rxn):
    adata = _make_adata(["A", "C", "D", "E"])
    report = generate_completeness_report(
        adata, simple_gpr, simple_task_by_rxn, write_to_obs=True, return_matrix=True
    )
    assert set(report.keys()) == {"dataset", "cell"}
    assert set(report["dataset"].keys()) == {
        "task_completeness",
        "reaction_completeness",
        "overall_summary",
    }
    assert set(report["cell"].keys()) == {"per_cell", "matrix_essential", "matrix_all"}
    # obs should have been populated.
    assert "completeness_essential" in adata.obs.columns
    assert "completeness_all" in adata.obs.columns
    # dataset-level agrees with standalone call.
    dataset_only = compute_dataset_completeness(adata, simple_gpr, simple_task_by_rxn)
    pd.testing.assert_frame_equal(
        report["dataset"]["task_completeness"], dataset_only["task_completeness"]
    )
