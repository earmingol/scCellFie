import ast
import cobra
import numpy as np


def gene_score(gene_expression, gene_threshold):
    return 5*np.log(1 + gene_expression/(gene_threshold+0.01)) # Added small value to threshold to avoid division by zero


def compute_gene_scores(adata, thresholds):
    genes = [g for g in thresholds.index if g in adata.var_names]

    X = adata[:, genes].X.toarray()
    _thresholds = thresholds.loc[genes, thresholds.columns[:1]] # Use only first column, to avoid issues

    gene_scores = gene_score(X, _thresholds.values.T)
    adata.layers['gene_scores'] = gene_scores


def compute_gpr_gene_score(gpr, gene_scores):
    '''Recursive parsing of gprs into lists of complexes and their scores.'''
    if isinstance(gpr, cobra.core.gene.GPR):
        return compute_gpr_gene_score(gpr.body, gene_scores)
    elif isinstance(gpr, ast.Name):
        return gene_scores.get(gpr.id, 0), gpr.id  # Returns a default score of 0 if not found
    elif isinstance(gpr, ast.BoolOp):
        op = gpr.op
        if isinstance(op, ast.Or):
            max_score = 0
            max_gene = None
            for value in gpr.values:
                score, gene = compute_gpr_gene_score(value, gene_scores)
                if score > max_score:  # Find the maximum score
                    max_score, max_gene = score, gene
            return max_score, max_gene  # Return the maximum score and corresponding gene
        elif isinstance(op, ast.And):
            min_score = float('inf')
            min_gene = None
            for value in gpr.values:
                score, gene = compute_gpr_gene_score(value, gene_scores)
                if score < min_score:  # Find the minimum score
                    min_score, min_gene = score, gene
            return min_score, min_gene  # Return the minimum score and corresponding gene