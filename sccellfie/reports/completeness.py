import numpy as np
import pandas as pd
from scipy.sparse import issparse

from sccellfie.preprocessing.gpr_rules import find_genes_gpr
from sccellfie.stats.ablation import (
    _coerce_gpr_source,
    compute_gene_ablation_impact,
    essential_genes_from_ablation,
)


# ---------------- internal helpers ----------------


def _ensure_gpr_parsed_and_genes(gpr_source):
    """Parse gpr_source to cobra.GPR dicts and collect gene sets per reaction."""
    gpr_parsed = _coerce_gpr_source(gpr_source)
    genes_per_reaction = {
        r_id: set(find_genes_gpr(gpr.to_string()))
        for r_id, gpr in gpr_parsed.items()
    }
    return gpr_parsed, genes_per_reaction


def _identity_task_by_rxn(reactions):
    """Build an identity-like task_by_rxn DataFrame so each reaction is its own task."""
    return pd.DataFrame(
        np.eye(len(reactions), dtype=int),
        index=list(reactions),
        columns=list(reactions),
    )


def _genes_for_task(task, task_by_rxn, genes_per_reaction):
    """All genes appearing in any reaction of `task`."""
    rxns_in_task = task_by_rxn.columns[task_by_rxn.loc[task] != 0]
    genes = set()
    for r in rxns_in_task:
        if r in genes_per_reaction:
            genes |= genes_per_reaction[r]
    return genes


def _impact_sum(impact_df, task_col, gene_subset):
    """Sum impact_df[task_col] over the genes in gene_subset that appear as rows."""
    if not gene_subset:
        return 0.0
    rows = [g for g in gene_subset if g in impact_df.index]
    if not rows:
        return 0.0
    return float(impact_df.loc[rows, task_col].sum())


# ---------------- dataset-level ----------------


def compute_dataset_completeness(
    adata,
    gpr_source,
    task_by_rxn,
    ablation_impact=None,
    reaction_impact=None,
    metric="fraction_zeroed",
    threshold=1.0,
    disable_pbar=True,
):
    """
    Evaluate dataset completeness relative to a metabolic task database, at both
    essential-gene and all-gene scopes, in a single pass.

    "Missing" at the dataset level means the gene symbol does not appear in
    `adata.var_names`, i.e. the assay does not cover it. This is a property of
    the dataset as a whole, not of any particular cell.

    Parameters
    ----------
    adata : AnnData
        The user's expression data; only `adata.var_names` is consulted.

    gpr_source : dict
        Either `{reaction_id: cobra.core.gene.GPR}` (as returned by
        `sccellfie.preprocessing.prepare_inputs.preprocess_inputs`) or
        `{reaction_id: str}` of raw GPR strings.

    task_by_rxn : pandas.DataFrame
        Rows are tasks, columns are reactions; non-zero where reaction participates in task.

    ablation_impact : dict of DataFrames, optional
        Output of `sccellfie.stats.compute_gene_ablation_impact` at the task level.
        If None, it is computed internally.

    reaction_impact : dict of DataFrames, optional
        Reaction-level ablation (each reaction treated as its own task). Computed
        internally when None.

    metric : str, optional (default: 'fraction_zeroed')
        Impact metric used when deriving essential-gene sets via
        `essential_genes_from_ablation`. One of 'rel_change', 'abs_change',
        'fraction_zeroed'.

    threshold : float, optional (default: 1.0)
        Threshold paired with `metric` for essential-gene derivation.

    disable_pbar : bool, optional (default: True)
        Forwarded to internal `compute_gene_ablation_impact` calls when impacts are
        computed here.

    Returns
    -------
    dict
        Three flat DataFrames with dual essential/all scopes as suffixed columns:
            - 'task_completeness'     : one row per task.
            - 'reaction_completeness' : one row per reaction.
            - 'overall_summary'       : single row with aggregate stats.
    """
    gpr_parsed, genes_per_reaction = _ensure_gpr_parsed_and_genes(gpr_source)
    all_db_genes = set().union(*genes_per_reaction.values())
    genes_present_in_adata = set(adata.var_names)

    # --- ablation impact at task level ---
    if ablation_impact is None:
        ablation_impact = compute_gene_ablation_impact(
            gpr_parsed, task_by_rxn, disable_pbar=disable_pbar
        )
    task_rel = ablation_impact["rel_change"]

    # --- ablation impact at reaction level (identity task_by_rxn) ---
    if reaction_impact is None:
        identity_tbr = _identity_task_by_rxn(list(gpr_parsed.keys()))
        reaction_impact = compute_gene_ablation_impact(
            gpr_parsed, identity_tbr, disable_pbar=disable_pbar
        )
    rxn_rel = reaction_impact["rel_change"]

    # --- essential gene sets per task and per reaction ---
    essential_per_task = essential_genes_from_ablation(
        ablation_impact, metric=metric, threshold=threshold
    )
    essential_per_reaction = essential_genes_from_ablation(
        reaction_impact, metric=metric, threshold=threshold
    )

    # --- task-level DataFrame ---
    task_rows = []
    for task in task_by_rxn.index:
        all_genes = _genes_for_task(task, task_by_rxn, genes_per_reaction)
        ess_genes = set(essential_per_task.get(task, []))

        for scope_label, scope_set in (("essential", ess_genes), ("all", all_genes)):
            _ = scope_label  # linter satisfaction
        # Build both scopes' numbers per task.
        row = {}
        for scope_label, scope_set in (("essential", ess_genes), ("all", all_genes)):
            missing = scope_set - genes_present_in_adata
            present = scope_set & genes_present_in_adata
            n_total = len(scope_set)
            n_present = len(present)
            n_missing = len(missing)
            frac_present = (n_present / n_total) if n_total > 0 else 1.0

            if task in task_rel.columns and scope_set:
                total_impact = _impact_sum(task_rel, task, missing)
            else:
                total_impact = 0.0
            impact_weighted = 1.0 - float(np.clip(total_impact, 0.0, 1.0))

            col_label = "n_genes" if scope_label == "all" else "n_genes"
            if scope_label == "essential":
                row["n_genes_essential"] = n_total
                row["n_present_essential"] = n_present
                row["n_missing_essential"] = n_missing
                row["fraction_present_essential"] = frac_present
                row["impact_weighted_completeness_essential"] = impact_weighted
                row["total_impact_missing_essential"] = total_impact
                row["missing_genes_essential"] = ";".join(sorted(missing))
            else:
                row["n_genes_all"] = n_total
                row["n_present_all"] = n_present
                row["n_missing_all"] = n_missing
                row["fraction_present_all"] = frac_present
                row["impact_weighted_completeness_all"] = impact_weighted
                row["total_impact_missing_all"] = total_impact
                row["missing_genes_all"] = ";".join(sorted(missing))
        task_rows.append(row)

    task_completeness = pd.DataFrame(task_rows, index=task_by_rxn.index)

    # --- reaction-level DataFrame ---
    rxn_rows = []
    rxn_index = list(task_by_rxn.columns)
    for rxn in rxn_index:
        all_genes = genes_per_reaction.get(rxn, set())
        ess_genes = set(essential_per_reaction.get(rxn, []))

        row = {}
        for scope_label, scope_set in (("essential", ess_genes), ("all", all_genes)):
            missing = scope_set - genes_present_in_adata
            present = scope_set & genes_present_in_adata
            n_total = len(scope_set)
            n_present = len(present)
            n_missing = len(missing)
            frac_present = (n_present / n_total) if n_total > 0 else 1.0

            if rxn in rxn_rel.columns and scope_set:
                total_impact = _impact_sum(rxn_rel, rxn, missing)
            else:
                total_impact = 0.0
            impact_weighted = 1.0 - float(np.clip(total_impact, 0.0, 1.0))

            if scope_label == "essential":
                row["n_genes_essential"] = n_total
                row["n_present_essential"] = n_present
                row["n_missing_essential"] = n_missing
                row["fraction_present_essential"] = frac_present
                row["impact_weighted_completeness_essential"] = impact_weighted
                row["total_impact_missing_essential"] = total_impact
                row["missing_genes_essential"] = ";".join(sorted(missing))
            else:
                row["n_genes_all"] = n_total
                row["n_present_all"] = n_present
                row["n_missing_all"] = n_missing
                row["fraction_present_all"] = frac_present
                row["impact_weighted_completeness_all"] = impact_weighted
                row["total_impact_missing_all"] = total_impact
                row["missing_genes_all"] = ";".join(sorted(missing))
        rxn_rows.append(row)

    reaction_completeness = pd.DataFrame(rxn_rows, index=rxn_index)

    # --- overall summary ---
    summary = {
        "n_tasks_total": len(task_completeness),
        "n_tasks_fully_covered_essential": int(
            (task_completeness["fraction_present_essential"] == 1.0).sum()
        ),
        "n_tasks_fully_covered_all": int(
            (task_completeness["fraction_present_all"] == 1.0).sum()
        ),
        "n_tasks_compromised_essential": int(
            (task_completeness["impact_weighted_completeness_essential"] < 1.0).sum()
        ),
        "n_tasks_compromised_all": int(
            (task_completeness["impact_weighted_completeness_all"] < 1.0).sum()
        ),
        "mean_task_impact_weighted_essential": float(
            task_completeness["impact_weighted_completeness_essential"].mean()
        ),
        "mean_task_impact_weighted_all": float(
            task_completeness["impact_weighted_completeness_all"].mean()
        ),
        "mean_task_fraction_present_essential": float(
            task_completeness["fraction_present_essential"].mean()
        ),
        "mean_task_fraction_present_all": float(
            task_completeness["fraction_present_all"].mean()
        ),
        "n_reactions_total": len(reaction_completeness),
        "n_reactions_fully_covered_essential": int(
            (reaction_completeness["fraction_present_essential"] == 1.0).sum()
        ),
        "n_reactions_fully_covered_all": int(
            (reaction_completeness["fraction_present_all"] == 1.0).sum()
        ),
        "n_genes_in_db_total": len(all_db_genes),
        "n_genes_in_db_present_in_adata": len(all_db_genes & genes_present_in_adata),
        "fraction_db_genes_present": (
            len(all_db_genes & genes_present_in_adata) / len(all_db_genes)
            if all_db_genes
            else 1.0
        ),
    }
    overall_summary = pd.DataFrame([summary])

    return {
        "task_completeness": task_completeness,
        "reaction_completeness": reaction_completeness,
        "overall_summary": overall_summary,
    }


# ---------------- per-cell ----------------


def _zero_mask(adata, gene_subset, layer=None):
    """(n_cells, n_selected_genes) float matrix where entry is 1.0 iff expression == 0."""
    genes_in_adata = [g for g in gene_subset if g in adata.var_names]
    if not genes_in_adata:
        return genes_in_adata, np.zeros((adata.n_obs, 0))
    sub = adata[:, genes_in_adata]
    X = sub.layers[layer] if layer is not None else sub.X
    if issparse(X):
        X = X.toarray()
    return genes_in_adata, (X == 0).astype(float)


def _per_cell_task_impact(
    adata, task, scope_set, impact_df, genes_present_in_adata, layer
):
    """
    Impact-weighted per-cell shortfall for a single (task, scope).

    Returns
    -------
    np.ndarray of shape (n_cells,)
        Per-cell `total_impact_missing` for the task. Caller applies `1 - clip(·, 0, 1)`.
    """
    if not scope_set:
        return np.zeros(adata.n_obs)

    # Dataset-absent genes contribute a constant per-cell shortfall.
    absent = scope_set - genes_present_in_adata
    if task in impact_df.columns and absent:
        absent_rows = [g for g in absent if g in impact_df.index]
        dataset_absent_impact = (
            float(impact_df.loc[absent_rows, task].sum()) if absent_rows else 0.0
        )
    else:
        dataset_absent_impact = 0.0

    # In-adata genes: shortfall contributed per cell iff expression is 0 in that cell.
    in_adata = scope_set & genes_present_in_adata
    if task not in impact_df.columns or not in_adata:
        return np.full(adata.n_obs, dataset_absent_impact)

    in_adata = [g for g in in_adata if g in impact_df.index]
    if not in_adata:
        return np.full(adata.n_obs, dataset_absent_impact)

    _, zero_mat = _zero_mask(adata, in_adata, layer=layer)
    impact_vec = impact_df.loc[in_adata, task].values.astype(float)
    cell_zero_impact = zero_mat @ impact_vec
    return cell_zero_impact + dataset_absent_impact


def compute_cell_completeness(
    adata,
    gpr_source,
    task_by_rxn,
    ablation_impact=None,
    metric="fraction_zeroed",
    threshold=1.0,
    layer=None,
    write_to_obs=True,
    obs_key_prefix="completeness_",
    return_matrix=False,
    disable_pbar=True,
):
    """
    Per-cell completeness relative to the metabolic-task database, at both essential-gene
    and all-gene scopes.

    "Missing" for a given cell and gene means either (a) the gene is absent from
    `adata.var_names` (dataset-absent, constant across cells) or (b) the gene is in
    `adata.var_names` but has expression `== 0` in that cell. Missing genes contribute
    their `rel_change` impact on each task; per-cell per-task completeness is
    `1 - clip(sum of impacts, 0, 1)`. The final per-cell score aggregates across tasks
    via the mean.

    Parameters
    ----------
    adata : AnnData
        Expression data. `adata.X` (or `adata.layers[layer]` if provided) is used to
        determine which genes are zero in which cells.

    gpr_source : dict
        As in `compute_dataset_completeness`.

    task_by_rxn : pandas.DataFrame
        Tasks x reactions membership matrix.

    ablation_impact : dict of DataFrames, optional
        Task-level ablation output. Computed internally if None.

    metric, threshold : see `compute_dataset_completeness`.

    layer : str, optional
        Layer in `adata.layers` from which to read expression. Defaults to `adata.X`.

    write_to_obs : bool, optional (default: True)
        If True, writes `adata.obs[obs_key_prefix + 'essential']` and
        `adata.obs[obs_key_prefix + 'all']`.

    obs_key_prefix : str, optional (default: 'completeness_')
        Prefix for the obs columns when `write_to_obs=True`.

    return_matrix : bool, optional (default: False)
        If True, also return the dense `(cell x task)` per-scope completeness matrices.

    disable_pbar : bool, optional (default: True)
        Forwarded to internal `compute_gene_ablation_impact` call when impact is
        computed here.

    Returns
    -------
    dict
        - 'per_cell'         : DataFrame(cells x ['completeness_essential', 'completeness_all'])
        - 'matrix_essential' : DataFrame(cells x tasks) or None
        - 'matrix_all'       : DataFrame(cells x tasks) or None
    """
    gpr_parsed, genes_per_reaction = _ensure_gpr_parsed_and_genes(gpr_source)
    genes_present_in_adata = set(adata.var_names)

    if ablation_impact is None:
        ablation_impact = compute_gene_ablation_impact(
            gpr_parsed, task_by_rxn, disable_pbar=disable_pbar
        )
    task_rel = ablation_impact["rel_change"]

    essential_per_task = essential_genes_from_ablation(
        ablation_impact, metric=metric, threshold=threshold
    )

    tasks = list(task_by_rxn.index)
    matrix_ess = np.ones((adata.n_obs, len(tasks)), dtype=float)
    matrix_all = np.ones((adata.n_obs, len(tasks)), dtype=float)

    for j, task in enumerate(tasks):
        all_genes = _genes_for_task(task, task_by_rxn, genes_per_reaction)
        ess_genes = set(essential_per_task.get(task, []))

        total_ess = _per_cell_task_impact(
            adata, task, ess_genes, task_rel, genes_present_in_adata, layer
        )
        total_all = _per_cell_task_impact(
            adata, task, all_genes, task_rel, genes_present_in_adata, layer
        )

        matrix_ess[:, j] = 1.0 - np.clip(total_ess, 0.0, 1.0)
        matrix_all[:, j] = 1.0 - np.clip(total_all, 0.0, 1.0)

    matrix_ess_df = pd.DataFrame(matrix_ess, index=adata.obs_names, columns=tasks)
    matrix_all_df = pd.DataFrame(matrix_all, index=adata.obs_names, columns=tasks)

    per_cell = pd.DataFrame(
        {
            "completeness_essential": matrix_ess_df.mean(axis=1),
            "completeness_all": matrix_all_df.mean(axis=1),
        },
        index=adata.obs_names,
    )

    if write_to_obs:
        adata.obs[obs_key_prefix + "essential"] = per_cell["completeness_essential"].values
        adata.obs[obs_key_prefix + "all"] = per_cell["completeness_all"].values

    return {
        "per_cell": per_cell,
        "matrix_essential": matrix_ess_df if return_matrix else None,
        "matrix_all": matrix_all_df if return_matrix else None,
    }


# ---------------- convenience wrapper ----------------


def generate_completeness_report(
    adata,
    gpr_source,
    task_by_rxn,
    ablation_impact=None,
    reaction_impact=None,
    metric="fraction_zeroed",
    threshold=1.0,
    layer=None,
    write_to_obs=True,
    obs_key_prefix="completeness_",
    return_matrix=False,
    disable_pbar=True,
):
    """
    Run both `compute_dataset_completeness` and `compute_cell_completeness` in one call
    and return `{'dataset': ..., 'cell': ...}`. Shares the same ablation impact across
    both sub-reports to avoid duplicate computation.
    """
    gpr_parsed, _ = _ensure_gpr_parsed_and_genes(gpr_source)
    if ablation_impact is None:
        ablation_impact = compute_gene_ablation_impact(
            gpr_parsed, task_by_rxn, disable_pbar=disable_pbar
        )

    dataset = compute_dataset_completeness(
        adata,
        gpr_parsed,
        task_by_rxn,
        ablation_impact=ablation_impact,
        reaction_impact=reaction_impact,
        metric=metric,
        threshold=threshold,
        disable_pbar=disable_pbar,
    )
    cell = compute_cell_completeness(
        adata,
        gpr_parsed,
        task_by_rxn,
        ablation_impact=ablation_impact,
        metric=metric,
        threshold=threshold,
        layer=layer,
        write_to_obs=write_to_obs,
        obs_key_prefix=obs_key_prefix,
        return_matrix=return_matrix,
        disable_pbar=disable_pbar,
    )
    return {"dataset": dataset, "cell": cell}
