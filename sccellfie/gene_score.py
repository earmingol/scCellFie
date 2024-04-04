import ast
import cobra
import numpy as np

from scipy.sparse import issparse
from sccellfie.tests import PCOUNT


def gene_score(gene_expression, gene_threshold):
    '''
    Computes the gene score for a given gene expression and threshold.

    Parameters
    ----------
    gene_expression: float or numpy.ndarray
        A float or a numpy array containing the gene expression values.

    gene_threshold: numpy.ndarray
        A float or a numpy array containing the gene thresholds.

    Returns
    -------
    gene_scores: float numpy.ndarray
        A float or a numpy array containing the gene scores.

    Notes
    -----
    This score is computed as previously indicated in the CellFie paper (https://doi.org/10.1016/j.cels.2019.05.012).
    '''
    return 5*np.log(1 + gene_expression/(gene_threshold + PCOUNT)) # Added small value to threshold to avoid division by zero


def compute_gene_scores(adata, thresholds, use_raw=False, layer='gene_scores'):
    '''
    Computes the gene scores from CellFie for each gene in an AnnData object given
    specific threshold values.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    thresholds: pandas.DataFrame
        A pandas.DataFrame object with the threshold values for each gene.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    layer: str, optional (default: 'gene_scores')
        The name of the layer in adata to store the gene scores.

    Returns
    -------
    None
        A numpy.ndarray object containing the gene scores is added to adata.layers[layer].

    Notes
    -----
    This score is computed as previously indicated in the CellFie paper (https://doi.org/10.1016/j.crmeth.2021.100040).
    '''
    genes = [g for g in thresholds.index if g in adata.var_names]
    if use_raw:
        X = adata[:, genes].raw.X
    else:
        X = adata[:, genes].X

    if issparse(X):
        X = X.toarray()

    _thresholds = thresholds.loc[genes, thresholds.columns[:1]] # Use only first column, to avoid issues

    gene_scores = gene_score(X, _thresholds.values.T)
    adata.layers[layer] = gene_scores


def compute_gpr_gene_score(gpr, gene_scores):
    '''
    Recursive parsing of gprs into lists of complexes and their scores.

    Parameters
    ----------
    gpr: cobra.core.gene.GPR or ast.Name or ast.BoolOp
        A GPR object or an abstract syntax tree (AST) object.

    gene_scores: dict
        A dictionary containing the gene scores for each gene. It's recommended
        to use a defaultdict with a default value of 0 for assigning a score to genes
        not found in the dictionary.

    Returns
    -------
    score: float
        The score of the gpr.

    gene: str
        The gene with the best score in the gpr. This score could be min or max, depending
        on the GPR (and, or).

    Notes
    -----
    This score is computed as previously indicated in the CellFie paper (https://doi.org/10.1016/j.crmeth.2021.100040).
    '''
    if isinstance(gpr, cobra.core.gene.GPR):
        return compute_gpr_gene_score(gpr.body, gene_scores)
    elif isinstance(gpr, ast.Name):
        return gene_scores.get(gpr.id, 0), gpr.id  # Returns a default score of 0 if not found
    elif isinstance(gpr, ast.BoolOp):
        op = gpr.op
        if isinstance(op, ast.Or):
            max_score = float('-inf')
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