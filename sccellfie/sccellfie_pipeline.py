import scanpy as sc
from tqdm import tqdm

from sccellfie.datasets.database import load_sccellfie_database
from sccellfie.io.save_data import save_adata
from sccellfie.preprocessing.adata_utils import normalize_adata, transform_adata_gene_names
from sccellfie.preprocessing.prepare_inputs import preprocess_inputs, CORRECT_GENES
from sccellfie.expression.smoothing import smooth_expression_knn
from sccellfie.gene_score import compute_gene_scores
from sccellfie.reaction_activity import compute_reaction_activity
from sccellfie.metabolic_task import compute_mt_score


def run_sccellfie_pipeline(adata, organism='human', sccellfie_data_folder=None, sccellfie_db=None, n_counts_col='n_counts',
                           process_by_group=False, groupby=None, neighbors_key='neighbors',n_neighbors=10, batch_key=None,
                           threshold_key='sccellfie_threshold', smooth_cells=True, alpha=0.33, chunk_size=5000,
                           disable_pbar=False, save_folder=None, save_filename=None, verbose=True):
    """
    Runs the complete scCellFie pipeline on the given AnnData object, processing by cell type if specified.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression values and nearest neighbor graph.
        The .X matrix must contain raw counts. If neighbors are not present, they will be computed.

    organism : str, optional (default: 'human')
        Organism for the analysis. Options are 'human' or 'mouse'.

    sccellfie_data_folder : str, optional (default: None)
        Path to the folder containing the files of the scCellFie database
        (reactions, GPR rules, metabolic tasks, etc.).

    sccellfie_db : dict, optional (default: None)
        Dictionary containing the scCellFie database information.
        If this information is provided, the sccellfie_data_folder will be ignored.
        This dictionary must contain the keys 'rxn_info', 'task_by_gene', 'rxn_by_gene', 'task_by_rxn', 'thresholds', and
        'organism'.
        Examples of dataframes can be found at
        https://github.com/earmingol/scCellFie/raw/refs/heads/main/task_data/homo_sapiens/

    n_counts_col : str, optional (default: 'n_counts')
        Column name in adata.obs containing the total counts per cell.
        If None or not present, the total counts will be computed.

    process_by_group : bool, optional (default: False)
        Whether to process data by groups (e.g., cell types). This is intended to be
        memory efficient for huge datasets. Results will be not outputted and will be
        saved to the disk instead. If True, `groupby` must be specified, as well as `save_folder`.

    groupby : str, optional (default: None)
        Column name in adata.obs for the groups to process. Required if process_by_group is True.

    neighbors_key : str, optional (default: 'neighbors')
        Key in adata.uns for neighbor data. If not present, neighbors will be computed.

    n_neighbors : int, optional (default: 10)
        Number of neighbors to find (if `neighbors_key` is not present). This number of neighbors
        will be used in the KNN smoothing too, if `smooth_cells` is True.

    batch_key : str, optional (default: None)
        Column name in adata.obs for batch information. If present, Harmony will be used to
        integrate the data before computing neighbors (when neighbors are not present in the AnnData object).

    threshold_key : str, optional (default: 'sccellfie_threshold')
        Key for the threshold to use in gene score computation.
        This key is present in the threshold file of the scCellFie (or custom) database.

    smooth_cells : bool, optional (default: True)
        Whether to perform a smoothing for the expression values based on the nearest neighbors.
        If True, KNN smoothing will be performed.

    alpha : float, optional (default: 0.33)
        The weight or fraction of the smoothed expression to use in the final expression matrix.
        The final expression matrix is computed as (1 - alpha) * X + alpha * (S @ X), where X is the
        original expression matrix and S is the smoothed matrix.

    chunk_size : int, optional (default: 5000)
        Size of chunks for smoothing the expression of large datasets.
        This is used to split the data into smaller parts to reduce memory usage.

    disable_pbar : bool, optional (default: False)
        Whether to disable the progress bar.

    save_folder : str, optional (default: None)
        Folder to save results. The AnnData object is saved to folder/save_filename.h5ad.
        The scCellFie attributes are saved to:
            - reactions: folder/save_filename_reactions.h5ad.
            - metabolic_tasks: folder/save_filename_metabolic_tasks.h5ad.
        If process_by_group is True, file names will include cell type name as in
         save_filename_celltype.h5ad.

    save_filename : str, optional (default: None)
        Filename to save results. If None, the filename will be 'sccellfie'.

    verbose : bool, optional (default: True)
        Whether to print messages during the processing.

    Returns
    -------
    preprocessed_db : dict
        Complete preprocessed database including the processed AnnData object and scCellFie attributes/results,
        normally stored as preprocessed_db['adata'], preprocessed_db['adata'].reactions,
        and preprocessed_db['adata'].metabolic_tasks.
        This output is returned only if `process_by_group` is False.
    """
    if verbose:
        print("\n==== scCellFie Pipeline: Initializing ====")
        print(f"Loading scCellFie database for organism: {organism}")
    else:
        disable_pbar = True

    # Load scCellFie database
    if sccellfie_db is None:
        sccellfie_db = load_sccellfie_database(organism=organism, task_folder=sccellfie_data_folder)

    # Check for ENSEMBL IDs and prepare transformation if necessary
    ensembl_ids = all([g.startswith('ENS') for g in adata.var_names])

    # Default filename
    if save_filename is None:
        save_filename = 'sccellfie'

    if process_by_group:
        if groupby is None:
            raise ValueError("groupby must be specified when process_by_group is True")
        if save_folder is None:
            raise ValueError("save_folder must be specified when process_by_group is True")

        if verbose:
            print(f"\n==== scCellFie Pipeline: Processing by groups ====")
            print(f"Using column: {groupby}")

        preprocessed_db = None
        met_genes = None
        first_group = True

        for celltype, df in tqdm(adata.obs.groupby(groupby), desc='Processing groups', disable=disable_pbar):
            adata_tmp = adata[df.index, :].copy()

            if first_group:
                preprocessed_db = process_chunk(adata=adata_tmp,
                                                sccellfie_db=sccellfie_db,
                                                n_counts_col=n_counts_col,
                                                smooth_cells=smooth_cells,
                                                alpha=alpha,
                                                chunk_size=chunk_size,
                                                threshold_key=threshold_key,
                                                disable_pbar=True,
                                                neighbors_key=neighbors_key,
                                                n_neighbors=n_neighbors,
                                                batch_key=batch_key,
                                                ensembl_ids=ensembl_ids,
                                                organism=organism,
                                                verbose=False,
                                                first_group=True)
                met_genes = list(preprocessed_db['adata'].var_names)
                first_group = False
            else:
                preprocessed_db = process_chunk(adata=adata_tmp,
                                                sccellfie_db=sccellfie_db,
                                                n_counts_col=n_counts_col,
                                                smooth_cells=smooth_cells,
                                                alpha=alpha,
                                                chunk_size=chunk_size,
                                                threshold_key=threshold_key,
                                                disable_pbar=True,
                                                neighbors_key=neighbors_key,
                                                n_neighbors=n_neighbors,
                                                batch_key=batch_key,
                                                ensembl_ids=ensembl_ids,
                                                organism=organism,
                                                verbose=False,
                                                first_group=False,
                                                preprocessed_db=preprocessed_db,
                                                met_genes=met_genes)

            # Save results if requested
            if save_folder:
                save_adata(adata=preprocessed_db['adata'],
                           output_directory=save_folder,
                           filename=save_filename + '_' + celltype.replace(' ', '_'),
                           verbose=False)
        if verbose:
            print("\n==== scCellFie Pipeline: Processing completed successfully ====")
    else:
        if verbose:
            print("\n==== scCellFie Pipeline: Processing entire dataset ====")
        preprocessed_db = process_chunk(adata=adata,
                                        sccellfie_db=sccellfie_db,
                                        n_counts_col=n_counts_col,
                                        smooth_cells=smooth_cells,
                                        alpha=alpha,
                                        chunk_size=chunk_size,
                                        threshold_key=threshold_key,
                                        disable_pbar=disable_pbar,
                                        neighbors_key=neighbors_key,
                                        n_neighbors=n_neighbors,
                                        batch_key=batch_key,
                                        ensembl_ids=ensembl_ids,
                                        organism=organism,
                                        verbose=verbose,
                                        first_group=True)

        # Save results if requested
        if save_folder:
            if verbose:
                print("\n==== scCellFie Pipeline: Saving results ====")
            save_adata(adata=preprocessed_db['adata'],
                       output_directory=save_folder,
                       filename=save_filename)
        if verbose:
            print("\n==== scCellFie Pipeline: Processing completed successfully ====")
        return preprocessed_db


def process_chunk(adata, sccellfie_db, n_counts_col, smooth_cells, alpha, chunk_size, threshold_key, disable_pbar,
                  neighbors_key, n_neighbors, batch_key, ensembl_ids, organism, first_group=True, preprocessed_db=None,
                  met_genes=None, verbose=True, ):
    """
    Processes a chunk of data (either a cell type or the entire dataset).

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix containing the expression data and nearest neighbor graph.

    sccellfie_db : dict
        Dictionary containing the scCellFie database information.
        Keys are 'rxn_info', 'task_by_gene', 'rxn_by_gene', 'task_by_rxn', and 'thresholds'.

    n_counts_col : str
        Column name in adata.obs containing the total counts per cell.

    smooth_cells : bool
        Whether to perform a smoothing for the expression values based on the nearest neighbors.

    alpha : float
        Smoothing factor for KNN smoothing.

    chunk_size : int
        Size of chunks for processing large datasets.

    threshold_key : str
        Key for the threshold to use in gene score computation.

    disable_pbar : bool
        Whether to disable the progress bar.

    neighbors_key : str
        Key in adata.uns for neighbor data.

    n_neighbors : int
        Number of neighbors to find (if `neighbors_key` is not present).

    batch_key : str or None
        Column name in adata.obs for batch information.
        This is used for Harmony integration if neighbors are not present.

    ensembl_ids : bool
        Whether the gene names are Ensembl IDs.

    organism : str
        Organism for the analysis. Options are 'human' or 'mouse'.

    first_group : bool, optional (default: True)
        Whether this is the first group to process.

    preprocessed_db : dict, optional (default: None)
        Dictionary containing the processed data from previous groups.

    met_genes : list, optional (default: None)
        List of genes to filter for subsequent groups.

    verbose : bool, optional (default: True)
        Whether to print messages during the processing.

    Returns
    -------
    preprocessed_db : dict
        Complete preprocessed database including the processed AnnData object and scCellFie attributes/results,
        normally stored as preprocessed_db['adata'], preprocessed_db['adata'].reactions,
        and preprocessed_db['adata'].metabolic_tasks.
    """
    if verbose:
        print("\n---- scCellFie Step: Preprocessing data ----")

    # Ensure adata is in memory
    if adata.isbacked:
        adata = adata.to_memory()

    # Preprocessing
    adata.layers['counts'] = adata.X.copy()
    should_normalize = True  # Default assumption
    if 'normalization' in adata.uns:
        if 'method' in adata.uns['normalization']:
            should_normalize = adata.uns['normalization']['method'] != 'total_counts'
    if should_normalize:
        normalize_adata(adata, n_counts_key=n_counts_col)

    # Check for presence of neighbors / Run this earlier to use HVGs
    if (smooth_cells) & (neighbors_key not in adata.uns.keys()):
        if verbose:
            print("\n---- scCellFie Step: Computing neighbors ----")
        compute_neighbors_pipeline(adata=adata, batch_key=batch_key, n_neighbors=n_neighbors,
                                   verbose=verbose)

    # Transform gene names if necessary
    if ensembl_ids:
        adata = transform_adata_gene_names(adata=adata, organism=organism, copy=False, drop_unmapped=True)

    if first_group:
        if verbose:
            print("\n---- scCellFie Step: Preparing inputs ----")
        # Prepare scCellFie inputs and filter reactions, tasks and genes
        preprocessed_db = dict()
        preprocessed_db['adata'], preprocessed_db['gpr_rules'], preprocessed_db['task_by_gene'], preprocessed_db[
            'rxn_by_gene'], preprocessed_db['task_by_rxn'] = preprocess_inputs(
            adata=adata,
            gpr_info=sccellfie_db['rxn_info'],
            task_by_gene=sccellfie_db['task_by_gene'],
            rxn_by_gene=sccellfie_db['rxn_by_gene'],
            task_by_rxn=sccellfie_db['task_by_rxn'],
            correction_organism=organism,
            verbose=verbose
        )

        for k, v in sccellfie_db.items():
            if k not in preprocessed_db.keys():
                preprocessed_db[k] = v
    else:
        correction_dict = CORRECT_GENES[organism]
        correction_dict = {k: v for k, v in correction_dict.items() if v in met_genes}
        adata.var.index = [correction_dict[g] if g in correction_dict.keys() else g for g in adata.var.index]
        # Filter genes for subsequent groups
        adata = adata[:, met_genes]
        preprocessed_db['adata'] = adata

    # Smoothing of gene expression
    if smooth_cells:
        if verbose:
            print("\n---- scCellFie Step: Smoothing gene expression ----")

        # Perform smoothing based on neighbors
        smooth_expression_knn(adata=preprocessed_db['adata'],
                              alpha=alpha,
                              mode='adjacency',
                              neighbors_key=neighbors_key,
                              chunk_size=chunk_size if preprocessed_db['adata'].shape[0] > 30000 else None,
                              disable_pbar=disable_pbar)
        preprocessed_db['adata'].X = preprocessed_db['adata'].layers['smoothed_X']

    # Compute gene scores
    if verbose:
        print("\n---- scCellFie Step: Computing gene scores ----")

    compute_gene_scores(adata=preprocessed_db['adata'],
                        thresholds=preprocessed_db['thresholds'][[threshold_key]])

    # Compute reaction activity
    if verbose:
        print("\n---- scCellFie Step: Computing reaction activity ----")
    compute_reaction_activity(adata=preprocessed_db['adata'],
                              gpr_dict=preprocessed_db['gpr_rules'],
                              use_specificity=True,
                              disable_pbar=disable_pbar)

    # Compute metabolic task activity
    if verbose:
        print("\n---- scCellFie Step: Computing metabolic task activity ----")
    compute_mt_score(adata=preprocessed_db['adata'],
                     task_by_rxn=preprocessed_db['task_by_rxn'],
                     verbose=verbose)

    return preprocessed_db


def compute_neighbors_pipeline(adata, batch_key, n_neighbors=10, verbose=True):
    """
    Computes neighbors for the AnnData object. In addition,
    finds the UMAP embeddings from these neighbors if not present.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix containing the expression data.

    batch_key : str or None
        Column name in adata.obs for batch information.
        This is used for running Harmony integration (must be installed)
        if neighbors are not present.

    n_neighbors : int, optional (default: 10)
        Number of neighbors to find.

    verbose : bool, optional (default: True)
        Whether to print messages during the processing.

    Returns
    -------
    None
        The neighbors are stored in adata.uns['neighbors'].
        UMAP embeddings are stored in adata.obsm['X_umap'] if not present.
    """
    bdata = adata.copy()
    if 'normalization' not in bdata.uns.keys():
        sc.pp.normalize_total(bdata, target_sum=1e4)
    sc.pp.log1p(bdata)
    sc.pp.highly_variable_genes(bdata, n_top_genes=2000, flavor='seurat_v3', batch_key=batch_key)
    sc.tl.pca(bdata)
    adata.obsm['X_pca'] = bdata.obsm['X_pca']
    if batch_key:
        try:
            sc.external.pp.harmony_integrate(bdata, batch_key, verbose)
            rep = 'X_pca_harmony'
            adata.obsm[rep] = bdata.obsm[rep]
        except:
            rep = 'X_pca'
    else:
        rep = 'X_pca'
    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors)
    if 'X_umap' not in adata.obsm.keys():
        sc.tl.umap(adata)