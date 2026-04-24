import warnings
from collections import defaultdict

import cobra
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from sccellfie.gene_score import compute_gpr_gene_score
from sccellfie.preprocessing.gpr_rules import find_genes_gpr


def _coerce_gpr_source(gpr_source):
    """
    Accept either a {reaction_id: cobra.core.gene.GPR} dict or a {reaction_id: str} dict
    of raw GPR strings and return a normalized {reaction_id: cobra.core.gene.GPR} dict.

    Reactions with empty / None GPR values are skipped (a single warning lists them).
    """
    parsed = {}
    empty = []
    for rxn_id, val in gpr_source.items():
        if val is None or (isinstance(val, str) and not val.strip()):
            empty.append(rxn_id)
            continue
        if isinstance(val, cobra.core.gene.GPR):
            parsed[rxn_id] = val
        elif isinstance(val, str):
            parsed[rxn_id] = cobra.core.gene.GPR().from_string(val)
        else:
            raise TypeError(
                f"gpr_source['{rxn_id}'] must be a cobra.core.gene.GPR or str, "
                f"got {type(val).__name__}"
            )
    if empty:
        warnings.warn(
            f"Skipping {len(empty)} reaction(s) with empty/None GPR: {empty[:5]}"
            + (" ..." if len(empty) > 5 else ""),
            UserWarning,
        )
    return parsed


def _compute_baseline_mts(gpr_parsed, task_by_rxn, uniform_score):
    """
    Walk every reaction's GPR with uniform scores and build the (task,) baseline MTS series.

    Returns
    -------
    baseline_ral : dict[str, float]
        Reaction -> RAL under uniform reference.
    baseline_mts : pd.Series
        Task -> MTS (mean of task's reaction RALs, following sccellfie.metabolic_task.compute_mt_score).
    reactions_per_gene : dict[str, set[str]]
        Gene -> set of reaction IDs whose GPR contains the gene.
    genes_per_reaction : dict[str, set[str]]
        Reaction -> set of genes appearing in its GPR.
    """
    genes_per_reaction = {}
    reactions_per_gene = defaultdict(set)
    baseline_ral = {}

    for rxn_id, gpr in gpr_parsed.items():
        genes = set(find_genes_gpr(gpr.to_string()))
        genes_per_reaction[rxn_id] = genes
        for g in genes:
            reactions_per_gene[g].add(rxn_id)
        scores = {g: uniform_score for g in genes}
        if not genes:
            # Edge case: GPR with no genes (shouldn't occur in shipped DBs).
            baseline_ral[rxn_id] = 0.0
            continue
        ral, _ = compute_gpr_gene_score(gpr, scores)
        baseline_ral[rxn_id] = ral

    # Filter task_by_rxn to reactions we have baseline RAL for.
    rxns_in_both = [r for r in task_by_rxn.columns if r in baseline_ral]
    tbr = task_by_rxn.loc[:, rxns_in_both]
    ral_vec = np.array([baseline_ral[r] for r in rxns_in_both], dtype=float)

    rxns_per_task = tbr.sum(axis=1).astype(float)
    task_sum = tbr.values @ ral_vec
    with np.errstate(divide="ignore", invalid="ignore"):
        mts_vals = np.where(rxns_per_task > 0, task_sum / rxns_per_task.values, 0.0)
    baseline_mts = pd.Series(mts_vals, index=tbr.index)
    return baseline_ral, baseline_mts, reactions_per_gene, genes_per_reaction


def compute_gene_ablation_impact(
    gpr_source,
    task_by_rxn,
    genes=None,
    uniform_score=1.0,
    disable_pbar=False,
):
    """
    Simulate single-gene ablation on a synthetic uniform-expression reference and measure
    per-task impact.

    For each gene, set its gene_score to 0 (leaving every other gene at `uniform_score`),
    re-evaluate every reaction whose GPR contains the gene, then recompute metabolic-task
    scores using the same arithmetic as `sccellfie.metabolic_task.compute_mt_score`.

    Parameters
    ----------
    gpr_source : dict
        Either `{reaction_id: cobra.core.gene.GPR}` (as returned by
        `sccellfie.preprocessing.prepare_inputs.preprocess_inputs`) or
        `{reaction_id: str}` of raw GPR strings (parsed internally via
        `cobra.core.gene.GPR().from_string`).

    task_by_rxn : pandas.DataFrame
        Rows are metabolic tasks, columns are reactions. Cell (T, r) is non-zero iff
        reaction r participates in task T.

    genes : list of str, optional (default: None)
        Subset of genes to ablate. Default uses the union of all genes across the GPRs.
        Genes not appearing in any GPR contribute an all-zero row.

    uniform_score : float, optional (default: 1.0)
        Positive score assigned to every non-ablated gene. Exposed mainly for testing;
        `rel_change` and `fraction_zeroed` are scale-invariant, while `abs_change`
        scales linearly with this value.

    disable_pbar : bool, optional (default: False)
        Disable the per-gene progress bar.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Three `(gene x task)` DataFrames keyed by:
            - 'rel_change'     : (baseline_mts - ablated_mts) / baseline_mts, in [0, 1].
                                 1.0 means the gene fully zeros the task under uniform reference.
            - 'abs_change'     : baseline_mts - ablated_mts.
            - 'fraction_zeroed': 1 iff ablated_mts == 0 and baseline_mts > 0, else 0.

    Notes
    -----
    Under a single-cell uniform reference every reaction's baseline RAL equals
    `5*log(1 + uniform_score/uniform_score) = 5*log(2)` when reached through
    `compute_gene_scores`, but here we call the GPR walker directly on gene scores
    (not raw expression), so baseline RAL and baseline MTS are both equal to
    `uniform_score` exactly (min/max of constant `uniform_score` values). This is a
    property of the walker, not of the `gene_score` transform.
    """
    if uniform_score <= 0:
        raise ValueError(f"uniform_score must be positive, got {uniform_score}")

    gpr_parsed = _coerce_gpr_source(gpr_source)
    baseline_ral, baseline_mts, reactions_per_gene, genes_per_reaction = (
        _compute_baseline_mts(gpr_parsed, task_by_rxn, uniform_score)
    )

    all_genes = sorted(reactions_per_gene.keys())
    if genes is None:
        genes_to_ablate = all_genes
    else:
        genes_to_ablate = list(genes)

    task_index = baseline_mts.index
    tbr_valid = task_by_rxn.loc[:, [r for r in task_by_rxn.columns if r in baseline_ral]]
    rxns_per_task = tbr_valid.sum(axis=1).astype(float)

    rel_df = pd.DataFrame(0.0, index=genes_to_ablate, columns=task_index)
    abs_df = pd.DataFrame(0.0, index=genes_to_ablate, columns=task_index)
    frz_df = pd.DataFrame(0.0, index=genes_to_ablate, columns=task_index)

    for g in tqdm(genes_to_ablate, disable=disable_pbar, desc="Gene ablation"):
        affected_rxns = reactions_per_gene.get(g, set()) & set(baseline_ral.keys())
        if not affected_rxns:
            continue

        # Recompute RAL for affected reactions only.
        new_ral = dict(baseline_ral)
        for r in affected_rxns:
            genes_in_r = genes_per_reaction[r]
            scores = {h: uniform_score for h in genes_in_r}
            scores[g] = 0.0
            ral, _ = compute_gpr_gene_score(gpr_parsed[r], scores)
            new_ral[r] = ral

        # Recompute MTS only for tasks touching at least one affected reaction.
        affected_tasks = tbr_valid.index[
            (tbr_valid.loc[:, list(affected_rxns)] != 0).any(axis=1)
        ]
        if len(affected_tasks) == 0:
            continue

        sub_tbr = tbr_valid.loc[affected_tasks, :]
        ral_vec = np.array([new_ral[r] for r in sub_tbr.columns], dtype=float)
        task_sum = sub_tbr.values @ ral_vec
        denom = rxns_per_task.loc[affected_tasks].values
        with np.errstate(divide="ignore", invalid="ignore"):
            new_mts = np.where(denom > 0, task_sum / denom, 0.0)
        new_mts = pd.Series(new_mts, index=affected_tasks)

        base = baseline_mts.loc[affected_tasks]
        abs_change = base - new_mts
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_change = np.where(base > 0, abs_change / base, 0.0)
        fraction_zeroed = ((new_mts == 0) & (base > 0)).astype(float)

        rel_df.loc[g, affected_tasks] = rel_change
        abs_df.loc[g, affected_tasks] = abs_change.values
        frz_df.loc[g, affected_tasks] = fraction_zeroed.values

    return {
        "rel_change": rel_df,
        "abs_change": abs_df,
        "fraction_zeroed": frz_df,
    }


def compute_reaction_topology_essentiality(
    task_by_rxn,
    cobra_model,
    task_endpoints,
    treat_reversible_as_bidirectional=True,
    ignore_metabolites=None,
):
    """
    For each task with a user-supplied `(start_metabolite, end_metabolite)`, flag reactions
    that are essential for connecting start -> end through the task's metabolite graph.

    The graph has metabolites as nodes and reactions as edges. For each reaction in the
    task that is present in the cobra Model, an edge is added from every substrate to
    every product. When `treat_reversible_as_bidirectional` is True, reversible reactions
    also contribute reverse edges. Optionally, metabolites in `ignore_metabolites` are
    excluded from the graph (useful for currency metabolites like ATP/ADP/H+/H2O).

    A reaction is essential iff removing all of its edges disconnects `start_met` from
    `end_met`. Tasks without a specified endpoint pair are skipped (their column in the
    output is all False).

    Parameters
    ----------
    task_by_rxn : pandas.DataFrame
        Rows are tasks, columns are reactions, non-zero where the reaction participates
        in the task.

    cobra_model : cobra.Model
        Genome-scale metabolic model whose reaction IDs and metabolite IDs match those
        used in `task_by_rxn`.

    task_endpoints : dict[str, tuple[str, str]]
        `{task_name: (start_metabolite_id, end_metabolite_id)}`. Only tasks listed here
        are evaluated; others get an all-False column.

    treat_reversible_as_bidirectional : bool, optional (default: True)
        If True, reactions with `rxn.reversibility == True` contribute edges in both
        directions. If False, edges follow the nominal substrate -> product direction only.

    ignore_metabolites : set of str, optional (default: None)
        Metabolite IDs to exclude as graph nodes. Edges that would use any of them are
        not added.

    Returns
    -------
    pandas.DataFrame
        (reactions x tasks) boolean DataFrame. `True` at (r, T) iff removing reaction r
        disconnects the start -> end path in task T. Rows are indexed by all reaction
        IDs in `task_by_rxn.columns`.
    """
    ignored = set(ignore_metabolites) if ignore_metabolites else set()
    reactions = list(task_by_rxn.columns)
    result = pd.DataFrame(False, index=reactions, columns=task_by_rxn.index)

    # Cache cobra lookups once.
    rxn_lookup = {}
    missing_ids = []
    for r_id in reactions:
        try:
            rxn_lookup[r_id] = cobra_model.reactions.get_by_id(r_id)
        except KeyError:
            missing_ids.append(r_id)
    if missing_ids:
        warnings.warn(
            f"{len(missing_ids)} reaction(s) present in task_by_rxn are missing from "
            f"the cobra Model and will be treated as non-essential by topology: "
            f"{missing_ids[:5]}" + (" ..." if len(missing_ids) > 5 else ""),
            UserWarning,
        )

    def _reaction_edges(rxn):
        """Return list of (src, dst) metabolite-id pairs contributed by this reaction."""
        subs = [m.id for m, c in rxn.metabolites.items() if c < 0 and m.id not in ignored]
        prods = [m.id for m, c in rxn.metabolites.items() if c > 0 and m.id not in ignored]
        edges = [(s, p) for s in subs for p in prods]
        if treat_reversible_as_bidirectional and rxn.reversibility:
            edges += [(p, s) for s in subs for p in prods]
        return edges

    def _build_graph(task_rxns, exclude=None):
        G = nx.DiGraph()
        for r_id in task_rxns:
            if exclude is not None and r_id == exclude:
                continue
            rxn = rxn_lookup.get(r_id)
            if rxn is None:
                continue
            for u, v in _reaction_edges(rxn):
                G.add_edge(u, v)
        return G

    skipped_tasks = []
    for task in task_by_rxn.index:
        if task not in task_endpoints:
            continue
        start_met, end_met = task_endpoints[task]
        task_rxns = [r for r in reactions if task_by_rxn.loc[task, r] != 0]

        G_full = _build_graph(task_rxns)
        if start_met not in G_full or end_met not in G_full:
            skipped_tasks.append((task, "endpoint metabolite missing from task subgraph"))
            continue
        if not nx.has_path(G_full, start_met, end_met):
            skipped_tasks.append((task, "no path from start to end in full task subgraph"))
            continue

        for r_id in task_rxns:
            if r_id not in rxn_lookup:
                continue  # absent from Model; leave as False
            G_r = _build_graph(task_rxns, exclude=r_id)
            if (
                start_met not in G_r
                or end_met not in G_r
                or not nx.has_path(G_r, start_met, end_met)
            ):
                result.loc[r_id, task] = True

    if skipped_tasks:
        warnings.warn(
            f"Topology not evaluated for {len(skipped_tasks)} task(s) "
            f"(missing endpoints or disconnected): "
            f"{[t for t, _ in skipped_tasks[:5]]}"
            + (" ..." if len(skipped_tasks) > 5 else ""),
            UserWarning,
        )

    return result


def essential_genes_from_ablation(
    impact,
    metric="fraction_zeroed",
    threshold=1.0,
    topology=None,
    task_by_rxn=None,
    gpr_source=None,
    fallback_to_ablation_only=True,
):
    """
    Derive per-task essential-gene lists from the ablation impact output, optionally
    filtered by a reaction-level topology essentiality DataFrame.

    Parameters
    ----------
    impact : dict[str, pandas.DataFrame] or pandas.DataFrame
        Output from `compute_gene_ablation_impact`, or one of its DataFrames.

    metric : str, optional (default: 'fraction_zeroed')
        Which impact DataFrame to threshold when `impact` is a dict. Must be one of
        `'rel_change'`, `'abs_change'`, `'fraction_zeroed'`.

    threshold : float, optional (default: 1.0)
        A gene is flagged essential for task T iff `impact[metric].loc[g, T] >= threshold`.

    topology : pandas.DataFrame, optional (default: None)
        `(reactions x tasks)` boolean DataFrame from `compute_reaction_topology_essentiality`.
        When provided, a gene is essential only if, in addition to clearing the threshold,
        at least one of the reactions it appears in (for that task) is marked essential by
        the topology.

    task_by_rxn : pandas.DataFrame, optional
        Required when `topology` is provided. Defines each task's reaction membership.

    gpr_source : dict, optional
        Required when `topology` is provided. Same format as
        `compute_gene_ablation_impact` (GPR objects or strings). Used to map genes to
        reactions.

    fallback_to_ablation_only : bool, optional (default: True)
        When `topology` is provided but a given task's column is all-False (not
        evaluated, no endpoints, or missing in the model): if True, fall back to the
        plain ablation threshold for that task. If False, yield [] for that task.

    Returns
    -------
    dict[str, list[str]]
        `{task_name: sorted list of essential genes}`.
    """
    if isinstance(impact, dict):
        if metric not in impact:
            raise ValueError(
                f"metric '{metric}' not in impact keys {list(impact.keys())}"
            )
        impact_df = impact[metric]
    elif isinstance(impact, pd.DataFrame):
        impact_df = impact
    else:
        raise TypeError(
            "impact must be a dict of DataFrames or a single DataFrame, "
            f"got {type(impact).__name__}"
        )

    if topology is not None:
        if task_by_rxn is None or gpr_source is None:
            raise ValueError(
                "topology filter requires both task_by_rxn and gpr_source"
            )
        gpr_parsed = _coerce_gpr_source(gpr_source)
        reactions_per_gene = defaultdict(set)
        for rxn_id, gpr in gpr_parsed.items():
            for g in set(find_genes_gpr(gpr.to_string())):
                reactions_per_gene[g].add(rxn_id)

    essential = {}
    for task in impact_df.columns:
        flagged_by_ablation = [
            g for g in impact_df.index if impact_df.loc[g, task] >= threshold
        ]

        if topology is None:
            essential[task] = sorted(flagged_by_ablation)
            continue

        if task not in topology.columns:
            if fallback_to_ablation_only:
                essential[task] = sorted(flagged_by_ablation)
            else:
                essential[task] = []
            continue

        topo_col = topology[task]
        if not topo_col.any():
            # Task not evaluated by topology.
            if fallback_to_ablation_only:
                essential[task] = sorted(flagged_by_ablation)
            else:
                essential[task] = []
            continue

        essential_rxns = set(topo_col.index[topo_col])
        # Restrict to reactions that actually belong to this task.
        task_rxns = set(task_by_rxn.columns[task_by_rxn.loc[task] != 0])
        essential_rxns &= task_rxns

        filtered = []
        for g in flagged_by_ablation:
            if reactions_per_gene.get(g, set()) & essential_rxns:
                filtered.append(g)
        essential[task] = sorted(filtered)

    return essential
