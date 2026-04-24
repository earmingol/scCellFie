import warnings

import cobra
import numpy as np
import pandas as pd
import pytest

from sccellfie.stats.ablation import (
    compute_gene_ablation_impact,
    compute_reaction_topology_essentiality,
    essential_genes_from_ablation,
)


# ---------- helpers ----------

def _gpr_strings_to_task(gpr_strings, task_by_rxn=None):
    """Run ablation on a dict of GPR strings; optionally with a task_by_rxn mapping.

    When task_by_rxn is None, one task per reaction (named 'task_<rxn>') with membership 1.
    """
    if task_by_rxn is None:
        rxn_ids = list(gpr_strings.keys())
        task_by_rxn = pd.DataFrame(
            np.eye(len(rxn_ids), dtype=int),
            index=[f"task_{r}" for r in rxn_ids],
            columns=rxn_ids,
        )
    return compute_gene_ablation_impact(gpr_strings, task_by_rxn, disable_pbar=True), task_by_rxn


def _build_cobra_model(reactions):
    """Build a tiny cobra Model from a list of (rxn_id, subs, prods, reversible) tuples.

    subs / prods are lists of metabolite IDs; reversible is bool.
    """
    model = cobra.Model("test")
    all_mets = set()
    for _, subs, prods, _ in reactions:
        all_mets.update(subs)
        all_mets.update(prods)
    for m_id in sorted(all_mets):
        model.add_metabolites([cobra.Metabolite(m_id)])
    for r_id, subs, prods, rev in reactions:
        rxn = cobra.Reaction(r_id)
        rxn.lower_bound = -1000.0 if rev else 0.0
        rxn.upper_bound = 1000.0
        coeffs = {}
        for m in subs:
            coeffs[model.metabolites.get_by_id(m)] = -1.0
        for m in prods:
            coeffs[model.metabolites.get_by_id(m)] = 1.0
        rxn.add_metabolites(coeffs)
        model.add_reactions([rxn])
    return model


# ---------- core ablation ----------

def test_pure_and():
    impact, _ = _gpr_strings_to_task({"R1": "A and B and C"})
    assert impact["rel_change"].loc["A", "task_R1"] == 1.0
    assert impact["rel_change"].loc["B", "task_R1"] == 1.0
    assert impact["rel_change"].loc["C", "task_R1"] == 1.0
    assert impact["fraction_zeroed"].loc["A", "task_R1"] == 1.0


def test_pure_or():
    impact, _ = _gpr_strings_to_task({"R1": "A or B or C"})
    for g in ["A", "B", "C"]:
        assert impact["rel_change"].loc[g, "task_R1"] == 0.0
        assert impact["fraction_zeroed"].loc[g, "task_R1"] == 0.0


def test_and_or_mixed():
    impact, _ = _gpr_strings_to_task({"R1": "(A and B) or C"})
    for g in ["A", "B", "C"]:
        assert impact["rel_change"].loc[g, "task_R1"] == 0.0
        assert impact["fraction_zeroed"].loc[g, "task_R1"] == 0.0


def test_or_of_and_with_shared_gene():
    impact, _ = _gpr_strings_to_task({"R1": "(A and B) or (A and C)"})
    assert impact["rel_change"].loc["A", "task_R1"] == 1.0
    assert impact["fraction_zeroed"].loc["A", "task_R1"] == 1.0
    assert impact["rel_change"].loc["B", "task_R1"] == 0.0
    assert impact["rel_change"].loc["C", "task_R1"] == 0.0


def test_subsumption_case():
    impact, _ = _gpr_strings_to_task({"R1": "A or (A and B)"})
    assert impact["rel_change"].loc["A", "task_R1"] == 1.0
    assert impact["rel_change"].loc["B", "task_R1"] == 0.0


def test_multi_reaction_task():
    gpr = {"R1": "A and B", "R2": "A or C"}
    task_by_rxn = pd.DataFrame(
        [[1, 1]], index=["T"], columns=["R1", "R2"]
    )
    impact, _ = _gpr_strings_to_task(gpr, task_by_rxn=task_by_rxn)
    # Baseline MTS = mean(1, 1) = 1
    # Ablate A: R1 = min(0,1)=0, R2 = max(0,1)=1 -> MTS=0.5 -> rel=0.5
    # Ablate B: R1 = 0, R2 = 1 -> MTS=0.5
    # Ablate C: R1 = 1, R2 = max(1,0)=1 -> MTS=1 -> rel=0
    assert impact["rel_change"].loc["A", "T"] == pytest.approx(0.5)
    assert impact["rel_change"].loc["B", "T"] == pytest.approx(0.5)
    assert impact["rel_change"].loc["C", "T"] == 0.0
    assert impact["fraction_zeroed"].loc["A", "T"] == 0.0
    assert impact["fraction_zeroed"].loc["B", "T"] == 0.0


def test_single_gene_gpr():
    impact, _ = _gpr_strings_to_task({"R1": "A"})
    assert impact["rel_change"].loc["A", "task_R1"] == 1.0
    assert impact["fraction_zeroed"].loc["A", "task_R1"] == 1.0


def test_duplicate_gene_gpr():
    impact, _ = _gpr_strings_to_task({"R1": "A and A"})
    assert impact["rel_change"].loc["A", "task_R1"] == 1.0
    assert impact["fraction_zeroed"].loc["A", "task_R1"] == 1.0


def test_dual_input_parity():
    gpr_strings = {"R1": "(A and B) or (A and C)"}
    task_by_rxn = pd.DataFrame([[1]], index=["T"], columns=["R1"])
    parsed = {k: cobra.core.gene.GPR().from_string(v) for k, v in gpr_strings.items()}

    impact_strings = compute_gene_ablation_impact(gpr_strings, task_by_rxn, disable_pbar=True)
    impact_parsed = compute_gene_ablation_impact(parsed, task_by_rxn, disable_pbar=True)
    for key in ["rel_change", "abs_change", "fraction_zeroed"]:
        pd.testing.assert_frame_equal(impact_strings[key], impact_parsed[key])


def test_gene_outside_gpr_is_noop():
    impact = compute_gene_ablation_impact(
        {"R1": "A and B"},
        pd.DataFrame([[1]], index=["T"], columns=["R1"]),
        genes=["A", "B", "Z"],
        disable_pbar=True,
    )
    assert (impact["rel_change"].loc["Z"] == 0.0).all()


def test_uniform_score_scale_invariance():
    gpr = {"R1": "A and B", "R2": "A or C"}
    tbr = pd.DataFrame([[1, 1]], index=["T"], columns=["R1", "R2"])
    imp1 = compute_gene_ablation_impact(gpr, tbr, uniform_score=1.0, disable_pbar=True)
    imp5 = compute_gene_ablation_impact(gpr, tbr, uniform_score=5.0, disable_pbar=True)
    pd.testing.assert_frame_equal(imp1["rel_change"], imp5["rel_change"])
    pd.testing.assert_frame_equal(imp1["fraction_zeroed"], imp5["fraction_zeroed"])
    # abs_change scales by 5x
    np.testing.assert_allclose(imp5["abs_change"].values, imp1["abs_change"].values * 5.0)


# ---------- topology ----------

def test_topology_linear_chain():
    model = _build_cobra_model([
        ("R1", ["A"], ["B"], False),
        ("R2", ["B"], ["C"], False),
        ("R3", ["C"], ["D"], False),
    ])
    tbr = pd.DataFrame([[1, 1, 1]], index=["T"], columns=["R1", "R2", "R3"])
    topo = compute_reaction_topology_essentiality(tbr, model, {"T": ("A", "D")})
    assert topo.loc["R1", "T"]
    assert topo.loc["R2", "T"]
    assert topo.loc["R3", "T"]


def test_topology_parallel_branch():
    model = _build_cobra_model([
        ("R1", ["A"], ["B"], False),
        ("R2a", ["B"], ["C"], False),
        ("R2b", ["B"], ["C"], False),
        ("R3", ["C"], ["D"], False),
    ])
    tbr = pd.DataFrame(
        [[1, 1, 1, 1]], index=["T"], columns=["R1", "R2a", "R2b", "R3"]
    )
    topo = compute_reaction_topology_essentiality(tbr, model, {"T": ("A", "D")})
    assert topo.loc["R1", "T"]
    assert topo.loc["R3", "T"]
    assert not topo.loc["R2a", "T"]
    assert not topo.loc["R2b", "T"]


def test_topology_chain_with_one_branch():
    model = _build_cobra_model([
        ("R1", ["A"], ["B"], False),
        ("R2", ["B"], ["C"], False),
        ("R3a", ["C"], ["D"], False),
        ("R3b", ["C"], ["D"], False),
    ])
    tbr = pd.DataFrame(
        [[1, 1, 1, 1]], index=["T"], columns=["R1", "R2", "R3a", "R3b"]
    )
    topo = compute_reaction_topology_essentiality(tbr, model, {"T": ("A", "D")})
    assert topo.loc["R1", "T"]
    assert topo.loc["R2", "T"]
    assert not topo.loc["R3a", "T"]
    assert not topo.loc["R3b", "T"]


def test_topology_reversibility():
    model = _build_cobra_model([
        ("R1", ["A"], ["B"], True),
        ("R2", ["B"], ["C"], True),
    ])
    tbr = pd.DataFrame([[1, 1]], index=["T"], columns=["R1", "R2"])
    # Bidirectional: path C -> A exists via reverse edges; both reactions essential.
    topo_bi = compute_reaction_topology_essentiality(
        tbr, model, {"T": ("C", "A")}, treat_reversible_as_bidirectional=True
    )
    assert topo_bi.loc["R1", "T"]
    assert topo_bi.loc["R2", "T"]

    # Forward-only: no path from C to A; task not evaluated.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        topo_fwd = compute_reaction_topology_essentiality(
            tbr, model, {"T": ("C", "A")}, treat_reversible_as_bidirectional=False
        )
    assert not topo_fwd["T"].any()
    assert any("not evaluated" in str(wi.message).lower() for wi in w)


def test_topology_ignore_metabolites():
    # A -> B via R1 and R2 (two alternative routes both using currency X).
    # If we don't ignore X, reactions share the X node and articulation analysis
    # looks different. Set up: R1: A + X -> B + X; R2 shortcut via X -> B is not a
    # separate path; instead test that removing R1 or R2 alone still leaves the
    # other available.
    model = _build_cobra_model([
        ("R1", ["A", "X"], ["B", "X"], False),
        ("R2", ["A", "X"], ["B", "X"], False),
    ])
    tbr = pd.DataFrame([[1, 1]], index=["T"], columns=["R1", "R2"])
    # Neither reaction alone is essential (either one connects A -> B).
    topo = compute_reaction_topology_essentiality(
        tbr, model, {"T": ("A", "B")}, ignore_metabolites={"X"}
    )
    assert not topo.loc["R1", "T"]
    assert not topo.loc["R2", "T"]


def test_topology_reaction_missing_from_model():
    model = _build_cobra_model([
        ("R1", ["A"], ["B"], False),
        ("R2", ["B"], ["C"], False),
    ])
    tbr = pd.DataFrame(
        [[1, 1, 1]], index=["T"], columns=["R1", "R2", "R_GHOST"]
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        topo = compute_reaction_topology_essentiality(tbr, model, {"T": ("A", "C")})
    assert not topo.loc["R_GHOST", "T"]
    assert topo.loc["R1", "T"]
    assert topo.loc["R2", "T"]
    assert any("missing from the cobra Model" in str(wi.message) for wi in w)


def test_topology_task_without_endpoints():
    model = _build_cobra_model([("R1", ["A"], ["B"], False)])
    tbr = pd.DataFrame([[1]], index=["T"], columns=["R1"])
    topo = compute_reaction_topology_essentiality(tbr, model, {})
    assert not topo["T"].any()


def test_topology_disconnected_endpoints():
    model = _build_cobra_model([
        ("R1", ["A"], ["B"], False),
        ("R2", ["C"], ["D"], False),
    ])
    tbr = pd.DataFrame([[1, 1]], index=["T"], columns=["R1", "R2"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        topo = compute_reaction_topology_essentiality(tbr, model, {"T": ("A", "D")})
    assert not topo["T"].any()
    assert any("not evaluated" in str(wi.message).lower() for wi in w)


# ---------- combined essential-gene derivation ----------

def test_essential_genes_ablation_only_and_cases():
    impact, _ = _gpr_strings_to_task({"R1": "A and B and C"})
    essential = essential_genes_from_ablation(impact, metric="fraction_zeroed", threshold=1.0)
    assert essential["task_R1"] == ["A", "B", "C"]


def test_essential_genes_ablation_only_or_cases():
    impact, _ = _gpr_strings_to_task({"R1": "A or B or C"})
    essential = essential_genes_from_ablation(impact, metric="fraction_zeroed", threshold=1.0)
    assert essential["task_R1"] == []


def test_essential_genes_rel_change_threshold():
    gpr = {"R1": "A and B", "R2": "A or C"}
    tbr = pd.DataFrame([[1, 1]], index=["T"], columns=["R1", "R2"])
    impact, _ = _gpr_strings_to_task(gpr, task_by_rxn=tbr)
    # rel_change: A=0.5, B=0.5, C=0. threshold=0.5 -> A, B flagged; C not.
    essential = essential_genes_from_ablation(
        impact, metric="rel_change", threshold=0.5
    )
    assert essential["T"] == ["A", "B"]


def test_essential_genes_with_topology_filter_parallel_branch():
    # R2a/R2b are replaceable -> not network-essential.
    model = _build_cobra_model([
        ("R1", ["A"], ["B"], False),
        ("R2a", ["B"], ["C"], False),
        ("R2b", ["B"], ["C"], False),
        ("R3", ["C"], ["D"], False),
    ])
    gpr = {"R1": "gA", "R2a": "gB", "R2b": "gC", "R3": "gD"}
    tbr = pd.DataFrame(
        [[1, 1, 1, 1]], index=["T"], columns=["R1", "R2a", "R2b", "R3"]
    )
    impact = compute_gene_ablation_impact(gpr, tbr, disable_pbar=True)
    topo = compute_reaction_topology_essentiality(tbr, model, {"T": ("A", "D")})
    # Each gene zeros exactly one of the task's 4 reactions -> rel_change = 0.25.
    essential = essential_genes_from_ablation(
        impact, metric="rel_change", threshold=0.25,
        topology=topo, task_by_rxn=tbr, gpr_source=gpr,
    )
    # gA -> R1 (essential), gD -> R3 (essential). gB/gC -> R2a/R2b (not essential).
    assert set(essential["T"]) == {"gA", "gD"}


def test_essential_genes_with_topology_filter_linear_chain():
    model = _build_cobra_model([
        ("R1", ["A"], ["B"], False),
        ("R2", ["B"], ["C"], False),
        ("R3a", ["C"], ["D"], False),
        ("R3b", ["C"], ["D"], False),
    ])
    gpr = {"R1": "gA", "R2": "gX", "R3a": "gY", "R3b": "gZ"}
    tbr = pd.DataFrame(
        [[1, 1, 1, 1]], index=["T"], columns=["R1", "R2", "R3a", "R3b"]
    )
    impact = compute_gene_ablation_impact(gpr, tbr, disable_pbar=True)
    topo = compute_reaction_topology_essentiality(tbr, model, {"T": ("A", "D")})
    essential = essential_genes_from_ablation(
        impact, metric="rel_change", threshold=0.25,
        topology=topo, task_by_rxn=tbr, gpr_source=gpr,
    )
    # gA and gX essential; gY/gZ are in replaceable R3a/R3b.
    assert set(essential["T"]) == {"gA", "gX"}


def test_essential_genes_fallback_flag():
    impact, tbr = _gpr_strings_to_task({"R1": "A and B"})
    # Topology with task column all-False (as if not evaluated).
    topo = pd.DataFrame(False, index=["R1"], columns=["task_R1"])
    essential_true = essential_genes_from_ablation(
        impact, metric="fraction_zeroed", threshold=1.0,
        topology=topo, task_by_rxn=tbr, gpr_source={"R1": "A and B"},
        fallback_to_ablation_only=True,
    )
    essential_false = essential_genes_from_ablation(
        impact, metric="fraction_zeroed", threshold=1.0,
        topology=topo, task_by_rxn=tbr, gpr_source={"R1": "A and B"},
        fallback_to_ablation_only=False,
    )
    assert essential_true["task_R1"] == ["A", "B"]
    assert essential_false["task_R1"] == []


# ---------- pipeline wiring ----------

def test_pipeline_compute_ablation_impact_flag():
    import scanpy as sc
    from sccellfie.sccellfie_pipeline import run_sccellfie_pipeline
    from sccellfie.datasets.toy_inputs import (
        create_controlled_adata,
        create_controlled_task_by_rxn,
        create_controlled_rxn_by_gene,
        create_controlled_task_by_gene,
        create_global_threshold,
        add_toy_neighbors,
    )

    adata = create_controlled_adata()
    adata = add_toy_neighbors(adata, n_neighbors=2)

    # Build a minimal sccellfie_db matching the toy shape.
    rxn_info = pd.DataFrame({
        "Reaction": ["rxn1", "rxn2", "rxn3", "rxn4"],
        "GPR-symbol": ["gene1", "gene2", "gene2 and gene3", "gene1 or gene3"],
    })
    thresholds = create_global_threshold(threshold=0.5, n_vars=adata.shape[1])
    thresholds.columns = ["sccellfie_threshold"]
    sccellfie_db = {
        "rxn_info": rxn_info,
        "task_info": pd.DataFrame(
            {"Task": ["task1", "task2", "task3", "task4"],
             "System": ["s"] * 4, "Subsystem": ["s"] * 4}
        ),
        "task_by_rxn": create_controlled_task_by_rxn(),
        "task_by_gene": create_controlled_task_by_gene(),
        "rxn_by_gene": create_controlled_rxn_by_gene(),
        "thresholds": thresholds,
        "organism": "human",
    }

    with_flag = run_sccellfie_pipeline(
        adata.copy(), organism="human", sccellfie_db=sccellfie_db,
        smooth_cells=False, compute_ablation_impact=True, verbose=False,
    )
    without_flag = run_sccellfie_pipeline(
        adata.copy(), organism="human", sccellfie_db=sccellfie_db,
        smooth_cells=False, compute_ablation_impact=False, verbose=False,
    )
    assert "ablation_impact" in with_flag
    assert "ablation_impact" not in without_flag
    imp = with_flag["ablation_impact"]
    assert set(imp.keys()) == {"rel_change", "abs_change", "fraction_zeroed"}
    assert isinstance(imp["rel_change"], pd.DataFrame)
