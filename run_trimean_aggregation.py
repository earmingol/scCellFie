#!/usr/bin/env python3
"""
Aggregate metabolic task scores per cell group using the trimean and save a CSV.

Reads a single metabolic_tasks .h5ad file (the ``_metabolic_tasks.h5ad`` sidecar
written by ``sccellfie.io.save_adata``, or any AnnData whose ``.X`` holds
metabolic-task scores), aggregates the scores within each group of ``--groupby``
using ``sccellfie.expression.agg_expression_cells`` (default ``trimean``), and
writes a (groups x tasks) CSV.

No normalization or re-scoring is performed — the metabolic-task scores in ``.X``
are aggregated as-is.

Example
-------
    python run_trimean_aggregation.py \
        --input   /path/to/MT_metabolic_tasks.h5ad \
        --output  /path/to/MT_trimean_by_celltype.csv \
        --groupby cell_type
"""

import os
import argparse
import logging

import numpy as np
import scanpy as sc

from sccellfie.expression.aggregation import agg_expression_cells

logging.basicConfig(level="INFO", format="[%(asctime)s][%(levelname)s] %(message)s")


def parse_args():
    p = argparse.ArgumentParser(
        description="Trimean (or other) aggregation of metabolic task scores per cell group."
    )
    p.add_argument("--input", required=True,
                   help="Path to the metabolic_tasks .h5ad file to aggregate.")
    p.add_argument("--output", required=True,
                   help="Path to the output .csv file (groups x tasks).")
    p.add_argument("--groupby", required=True,
                   help="adata.obs column to group cells by (e.g. cell_type, condition).")
    p.add_argument("--agg_func", default="trimean",
                   help="Aggregation function passed to agg_expression_cells (default: trimean). "
                        "Options: mean, median, 25p, 75p, trimean, topmean.")
    p.add_argument("--layer", default=None,
                   help="Layer to aggregate instead of adata.X (default: adata.X).")
    p.add_argument("--exclude_zeros", action="store_true",
                   help="Exclude zeros when aggregating (treated as NaN).")
    p.add_argument("--chunk_size", type=int, default=None,
                   help="Number of tasks (genes) to densify at a time, to bound peak memory "
                        "for percentile-based aggregations like trimean. Results are identical "
                        "to the unchunked path (default: None = whole matrix at once).")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "float64"],
                   help="Cast dense blocks to this dtype before aggregating to reduce memory "
                        "(default: None = keep source dtype).")
    return p.parse_args()


def main():
    args = parse_args()

    logging.info(f"Reading metabolic task scores from '{args.input}'")
    adata = sc.read_h5ad(args.input)
    logging.info(f"Loaded AnnData: {adata.n_obs:,} cells x {adata.n_vars:,} tasks")

    if args.groupby not in adata.obs.columns:
        raise KeyError(
            f"groupby column '{args.groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    n_groups = adata.obs[args.groupby].nunique()
    logging.info(
        f"Aggregating with '{args.agg_func}' over '{args.groupby}' "
        f"({n_groups} groups)"
        + (f", layer='{args.layer}'" if args.layer else "")
        + (", excluding zeros" if args.exclude_zeros else "")
        + (f", chunk_size={args.chunk_size}" if args.chunk_size else "")
        + (f", dtype={args.dtype}" if args.dtype else "")
    )

    # Only pass the newer kwargs when set, so this stays compatible with older
    # installs of sccellfie that predate `chunk_size` / `dtype`.
    kwargs = dict(
        groupby=args.groupby,
        agg_func=args.agg_func,
        layer=args.layer,
        exclude_zeros=args.exclude_zeros,
    )
    if args.chunk_size is not None:
        kwargs["chunk_size"] = args.chunk_size
    if args.dtype is not None:
        kwargs["dtype"] = np.dtype(args.dtype)

    # Returns a (groups x tasks) DataFrame: rows = groups, columns = task names.
    agg_df = agg_expression_cells(adata, **kwargs)
    agg_df.index.name = args.groupby
    logging.info(f"Aggregated matrix: {agg_df.shape[0]} groups x {agg_df.shape[1]} tasks")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    agg_df.to_csv(args.output)
    logging.info(f"Saved aggregated scores to '{args.output}'")


if __name__ == "__main__":
    main()
