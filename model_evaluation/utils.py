"""
Utility functions for evaluating cell type classification models.

This module provides functions to:
- Load and prepare cell data from CellXGene census
- Evaluate different model architectures (TabNet, Linear, MLP, CellTypist)
- Process and analyze classification results
- Handle data streaming and memory management
"""

import sys
import os
os.chdir('../scTab')
sys.path.append('.')
sys.path.append('../model_evaluation')
import numpy as np
import cellxgene_census
from scipy.sparse import csc_matrix
from cellnet.utils.data_loading import streamline_count_matrix, dataloader_factory
from cellnet.models import TabNet, MLP
from collections import OrderedDict
import scanpy as sc
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import csr_matrix
from torch import nn
import anndata
import celltypist
import os
import gc
from tqdm.auto import tqdm

def data_preparation(dataset_ids, features_file, var_file, cell_type_mapping_file, 
                    census_version='2023-05-15', filename=None, force_download=False, 
                    output_root=None):
    """
    Prepares data for model evaluation by downloading and processing cell data.

    Args:
        dataset_ids (str or list): Single dataset ID or list of dataset IDs to process
        features_file (str): Path to features.parquet containing gene information
        var_file (str): Path to var.parquet containing model variables
        cell_type_mapping_file (str): Path to cell type mapping file
        census_version (str): CellXGene census version to use
        filename (str, optional): Custom filename for output
        force_download (bool): Whether to force re-download existing data
        output_root (str): Root directory for storing AnnData chunks

    Returns:
        tuple: (output_folder, genes_from_model, cell_type_mapping)
    """
    # Load necessary reference files
    genes_from_model = pd.read_parquet(var_file)
    cell_type_mapping = pd.read_parquet(cell_type_mapping_file)
    genes_initial = pd.read_parquet(features_file)
    print("Setting up obs object")
    
    # Define supported 10x assay types
    assays = [
        "10x 5' v2", 
        "10x 3' v3", 
        "10x 3' v2", 
        "10x 5' v1", 
        "10x 3' v1", 
        "10x 3' transcription profiling", 
        "10x 5' transcription profiling"
    ]
    
    # Use dataset ID as filename if not specified
    if isinstance(dataset_ids, str) and filename == None:
        filename = dataset_ids
    
    # Handle special case for dataset differences
    if dataset_ids == 'diff_2023-05-15':
        dataset_ids = census_datasets_diff(census_version, '2023-05-15')
    
    # Extract all possible cell types from mapping
    cell_types = cell_type_mapping['label'].values.flatten().tolist()
    
    print("Creating AnnData object")
    
    # Validate and set filename
    if len(dataset_ids) == 1 and filename == None:
        filename = dataset_ids[0]
    if filename == None:
        raise ValueError("Please provide a filename for the dataset.")
    
    # Connect to census and filter observations
    census = cellxgene_census.open_soma(census_version=census_version)
    obs = (
        census["census_data"]["homo_sapiens"]
        .obs
        .read(
            column_names=["soma_joinid"],
            # Filter for primary data, matching datasets, cell types and assays
            value_filter=f"is_primary_data == True and dataset_id in {dataset_ids} and cell_type in {cell_types} and assay in {assays}"
        )
        .concat()
        .to_pandas()
    )
    
    # Create chunked AnnData objects and return paths
    output_folder = create_anndata_objects(
        obs, genes_initial, census, filename, 
        census_version, force_download, 131072, 
        output_root=output_root
    )
    return output_folder, genes_from_model, cell_type_mapping

def correct_labels(y_true: np.ndarray, y_pred: np.ndarray, child_matrix: np.ndarray):
    """
    Corrects predicted labels based on cell type hierarchy relationships.

    If a prediction is a child node of the true label, updates prediction to the true label.
    Example: If true label is 'T cell' and prediction is 'CD8 positive T cell',
    updates prediction to 'T cell'.

    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels  
        child_matrix: Matrix defining parent-child relationships

    Returns:
        np.ndarray: Array of corrected predictions
    """
    updated_predictions = y_pred.copy()
    # precalculate child nodes
    child_nodes = {i: np.where(child_matrix[i, :])[0] for i in range(child_matrix.shape[0])}

    for i, (pred, true_label) in enumerate(zip(y_pred, y_true)):
        if pred in child_nodes[true_label]:
            updated_predictions[i] = true_label
        else:
            updated_predictions[i] = pred

    return updated_predictions

def census_datasets_diff(new_census_version, old_census_version):
    """
    Compares two CellXGene census versions and returns added datasets.

    Args:
        new_census_version (str): New census version
        old_census_version (str): Old census version to compare against

    Returns:
        list: Dataset IDs present in new version but not in old version
    """
    census_new = cellxgene_census.open_soma(census_version=new_census_version)
    census_old = cellxgene_census.open_soma(census_version=old_census_version)
    datasets_new = census_new['census_info']['datasets'].read().concat().to_pandas()['dataset_id'].values.tolist()
    datasets_old = census_old['census_info']['datasets'].read().concat().to_pandas()['dataset_id'].values.tolist()
    return list(set(datasets_new) - set(datasets_old))

def create_anndata_objects(obs, genes_initial, census, filename, census_version, 
                         force_download=False, chunk_size=131072, output_root=None):
    """
    Creates AnnData objects from census data in chunks to manage memory.

    Args:
        obs: Observation data
        genes_initial: Initial gene features
        census: Census object
        filename: Base filename for chunks
        census_version: Census version
        force_download: Whether to force regeneration of existing chunks
        chunk_size: Number of cells per chunk
        output_root: Root directory for storing AnnData chunks

    Returns:
        str: Path to folder containing chunk files
    """
    if output_root:
        output_folder = os.path.join(output_root, f"{filename}_{census_version}_chunks")
    else:
        output_folder = f"./{filename}_{census_version}_chunks"
    os.makedirs(output_folder, exist_ok=True)
    protein_coding_genes = genes_initial.gene_names.tolist()
    total_obs = len(obs)
    for idx, start in enumerate(range(0, total_obs, chunk_size)):
        end = min(start + chunk_size, total_obs)
        chunk_file = os.path.join(output_folder, f"chunk_{idx}.h5ad")
        if not (os.path.exists(chunk_file)) or (force_download):
            chunk_coords = obs.soma_joinid.tolist()[start:end]
            adata_chunk = cellxgene_census.get_anndata(
                census=census,
                organism="Homo sapiens",
                X_name='raw',
                obs_coords=chunk_coords,
                var_value_filter=f"feature_name in {protein_coding_genes}",
                column_names={"obs": ["soma_joinid", "cell_type", "assay", "dataset_id", 
                                     "donor_id", "development_stage", "disease", 
                                     "tissue", "tissue_general"], 
                             "var": ['feature_id', 'feature_name']},
            )
            adata_chunk.raw = adata_chunk.copy()
            sc.pp.normalize_total(adata_chunk, target_sum=10000)
            sc.pp.log1p(adata_chunk)
            adata_chunk.write_h5ad(chunk_file)
            del adata_chunk
            manage_memory()
    return output_folder

def manage_memory():
    """Cleans up memory by running garbage collection and clearing CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

def streamline_data(adata, genes_from_model, cell_type_mapping):
    """
    Prepares AnnData object for model input by filtering and reordering features.

    Args:
        adata: AnnData object to process
        genes_from_model: Reference genes from model
        cell_type_mapping: Cell type mapping information

    Returns:
        Matrix: Processed expression matrix
    """
    # Filter cells by valid cell types
    cell_type_list = cell_type_mapping['label'].tolist()
    adata = adata[adata.obs['cell_type'].isin(cell_type_list)]
    
    # Reorder genes to match model's expected order
    genes_from_new_dataset_index = adata.var.feature_name.tolist().index
    reorder_ids = [genes_from_new_dataset_index(gene) for gene in genes_from_model.feature_name]
    x_streamlined = adata.X[:, reorder_ids]
    return x_streamlined

def run_tabnet(loader_gen, checkpoint_path, hparams_file, cell_type_hierarchy_file):
    '''
    loader, y_true: the return values from data_preparation
    checkpoint_path (str): file path to the specific .ckpt file 
    hparams_file (str): file path to the hparams.yaml file
    cell_type_hierarchy_file (str): file path to cell_type_hierarchy/child_matrix.npy
    '''
    ckpt = torch.load(checkpoint_path,)
    tabnet_weights = OrderedDict()
    for name, weight in ckpt['state_dict'].items():
        if 'classifier.' in name:
            tabnet_weights[name.replace('classifier.', '')] = weight

    # load in hparams file of model to get model architecture
    with open(hparams_file) as f:
        model_params = yaml.full_load(f.read())

    # initialzie model with hparams from hparams.yaml file
    tabnet = TabNet(
        input_dim=model_params['gene_dim'],
        output_dim=model_params['type_dim'],
        n_d=model_params['n_d'],
        n_a=model_params['n_a'],
        n_steps=model_params['n_steps'],
        gamma=model_params['gamma'],
        n_independent=model_params['n_independent'],
        n_shared=model_params['n_shared'],
        epsilon=model_params['epsilon'],
        virtual_batch_size=model_params['virtual_batch_size'],
        momentum=model_params['momentum'],
        mask_type=model_params['mask_type'],
    )

    # load trained weights
    tabnet.load_state_dict(tabnet_weights)
    # set model to inference mode
    tabnet.eval();
    return evaluate_model(tabnet, loader_gen, cell_type_hierarchy_file)

def run_linear(loader_gen, checkpoint_path, hparams_file, cell_type_hierarchy_file):
    '''
    loader, y_true: the return values from data_preparation
    checkpoint_path (str): file path to the specific .ckpt file 
    hparams_file (str): file path to the hparams.yaml file
    cell_type_hierarchy_file (str): file path to cell_type_hierarchy/child_matrix.npy
    '''
    #make option of what type of model you want to run 
    ckpt = torch.load(checkpoint_path)
    weights = OrderedDict()
    for name, weight in ckpt['state_dict'].items():
        if 'classifier.' in name:
            weights[name.replace('classifier.', '')] = weight

    # load in hparams file of model to get model architecture
    with open(hparams_file) as f:
        model_params = yaml.full_load(f.read())

    # initialzie model with hparams from hparams.yaml file
    linear = nn.Linear(model_params['gene_dim'], model_params['type_dim'])

    # load trained weights
    linear.load_state_dict(weights)
    # set model to inference mode
    linear.eval();
    return evaluate_model(linear, loader_gen, cell_type_hierarchy_file)

def run_mlp(loader_gen, checkpoint_path, hparams_file, cell_type_hierarchy_file):
    '''
    loader, y_true: the return values from data_preparation
    checkpoint_path (str): file path to the specific .ckpt file 
    hparams_file (str): file path to the hparams.yaml file
    cell_type_hierarchy_file (str): file path to cell_type_hierarchy/child_matrix.npy
    '''
    #make option of what type of model you want to run 
    #make option of what type of model you want to run and include helper functions
    ckpt = torch.load(checkpoint_path)
    weights = OrderedDict()
    for name, weight in ckpt['state_dict'].items():
        if 'classifier.' in name:
            weights[name.replace('classifier.', '')] = weight

    # load in hparams file of model to get model architecture
    with open(hparams_file) as f:
        model_params = yaml.full_load(f.read())

    # initialzie model with hparams from hparams.yaml file
    mlp = MLP(
        input_dim=model_params['gene_dim'],
        output_dim=model_params['type_dim'],
        hidden_size=model_params['hidden_size'],
        n_hidden=model_params['n_hidden'],
        dropout=model_params['dropout']
    )

    # load trained weights
    mlp.load_state_dict(weights)
    # set model to inference mode
    mlp.eval();
    return evaluate_model(mlp, loader_gen, cell_type_hierarchy_file)

def run_celltypist(loader_generator, checkpoint_path, cell_type_hierarchy_file):
    '''
    loader_gen: Generator yielding (adata, y_true) tuples from the prepared data.
    checkpoint_path: Path to the CellTypist model.
    cell_type_hierarchy_file: Path to the cell type hierarchy file.
    '''
    y_pred = [] 
    y_true_concatenated = []
    all_y_probs = []
    all_metadata = []

    for adata, _, y_true in tqdm(loader_generator, desc="Evaluting CellTypist"):
        if adata.raw is None:
            raise ValueError("The `.raw` attribute is missing in AnnData object.")
        x = adata.X
        var = adata.var.set_index("feature_name")
        
        # Collect metadata from AnnData object
        metadata = adata.obs[['cell_type', 'soma_joinid', 'assay', 'dataset_id', 
                             'donor_id', 'development_stage', 'disease', 
                             'tissue', 'tissue_general']].copy()
        all_metadata.append(metadata)
        
        preds = []
        probs_list = []

        for idxs in tqdm(np.array_split(np.arange(x.shape[0]), 10), desc="Batch", leave=False):
            x_batch = x[idxs, :]
            adata_test = anndata.AnnData(X=x_batch.todense(), var=var)
            adata_test.raw = adata_test.copy()
            pred_result = celltypist.annotate(adata_test, model=checkpoint_path, majority_voting=False)
            preds.append(pred_result)
            
            # Extract probability matrix
            probs_list.append(pred_result.probability_matrix.values)
            
            manage_memory() 
        
        chunk_y_pred = np.concatenate([batch.predicted_labels.to_numpy().flatten() for batch in preds])
        chunk_y_probs = np.vstack(probs_list)
        
        y_pred.append(chunk_y_pred)
        all_y_probs.append(chunk_y_probs)
        y_true_concatenated.extend(y_true)
        manage_memory()

    y_pred = np.concatenate(y_pred)
    y_true_concatenated = np.array(y_true_concatenated)
    all_y_probs = np.vstack(all_y_probs)
    all_metadata = pd.concat(all_metadata, ignore_index=True)

    cell_type_hierarchy = np.load(cell_type_hierarchy_file)
    y_pred_corr = correct_labels(y_true_concatenated, y_pred, cell_type_hierarchy)
    clf_report = pd.DataFrame(classification_report(y_true_concatenated, y_pred_corr, labels=np.unique(y_true_concatenated), output_dict=True)).T
    clf_report_overall = clf_report.iloc[-3:].copy()
    clf_report_per_class = clf_report.iloc[:-3].copy()
    
    return clf_report_overall, clf_report_per_class, all_y_probs, y_pred, y_true_concatenated, all_metadata

def evaluate_model(model, loader_generator, cell_type_hierarchy_file):
    '''
    loader, y_true: the return values from data_preparation
    cell_type_hierarchy_file (str): file path to cell_type_hierarchy/child_matrix.npy
    '''
    y_true_concatenated = []
    all_y_pred = []
    all_y_probs = []
    all_metadata = []

    for adata, loader, y_true in tqdm(loader_generator, desc="Evaluating model"):
        y_pred = []
        y_probs = []
        
        # Collect metadata from AnnData object
        metadata = adata.obs[['cell_type', 'soma_joinid', 'assay', 'dataset_id', 
                             'donor_id', 'development_stage', 'disease', 
                             'tissue', 'tissue_general']].copy()
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(loader), desc="Batch", leave=False):
                x_input = batch[0]['X']
                if isinstance(model, TabNet):
                    logits, _ = model(x_input)
                else:
                    logits = model(x_input)
                
                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                y_pred.append(torch.argmax(logits, dim=1).numpy())
                y_probs.append(probs.numpy())
                
                if i % 64 == 0:
                    manage_memory()
                    
        y_true_concatenated.extend(y_true)
        all_y_pred.extend(np.hstack(y_pred))
        all_y_probs.extend(np.vstack(y_probs))
        all_metadata.append(metadata)
        manage_memory()

    # Convert to numpy arrays
    y_true_concatenated = np.array(y_true_concatenated)
    all_y_pred = np.array(all_y_pred)
    all_y_probs = np.vstack(all_y_probs)
    all_metadata = pd.concat(all_metadata, ignore_index=True)

    # Compute classification metrics
    cell_type_hierarchy = np.load(cell_type_hierarchy_file)
    all_y_pred_corr = correct_labels(y_true_concatenated, all_y_pred, cell_type_hierarchy)
    clf_report = pd.DataFrame(classification_report(y_true_concatenated, all_y_pred_corr, labels=np.unique(y_true_concatenated), output_dict=True)).T
    clf_report_overall = clf_report.iloc[-3:].copy()
    clf_report_per_class = clf_report.iloc[:-3].copy()

    return clf_report_overall, clf_report_per_class, all_y_probs, all_y_pred, y_true_concatenated, all_metadata

def loader_generator(output_folder, genes_from_model, cell_type_mapping):
    """Generator that yields processed data chunks for model evaluation"""
    
    # Get all chunk files
    chunk_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".h5ad")]
    
    for chunk_file in sorted(chunk_files):
        # Load and process chunk
        adata = sc.read_h5ad(chunk_file)
        x_streamlined = streamline_data(adata, genes_from_model, cell_type_mapping)
        
        # Create numeric indices for cell types
        inverse_mapping = (
            cell_type_mapping
            .assign(idx=range(len(cell_type_mapping)))
            .set_index('label', drop=True)
        )
        
        # Convert cell type labels to numeric indices
        y_true = adata.obs.cell_type.tolist()
        y_true = inverse_mapping.loc[y_true]['idx'].to_numpy()
        
        # Create data loader with fixed batch size
        loader = dataloader_factory(x_streamlined, batch_size=2048, shuffle=False)
        yield adata, loader, y_true
        
        # Clean up memory
        del adata, x_streamlined, loader
        manage_memory()

def run_model(model, checkpoint_path, hparams_file, cell_type_hierarchy_file, genes_from_model, cell_type_mapping, output_folder):
    '''
    model (str): denoting which model to run ('tabnet', 'linear', 'mlp)
    '''
    loader_gen = loader_generator(output_folder, genes_from_model, cell_type_mapping)
    if model == 'tabnet':
        return run_tabnet(loader_gen, checkpoint_path, hparams_file, cell_type_hierarchy_file)
    if model == 'linear':
        return run_linear(loader_gen, checkpoint_path, hparams_file, cell_type_hierarchy_file)
    if model == 'mlp':
        return run_mlp(loader_gen, checkpoint_path, hparams_file, cell_type_hierarchy_file)
    if model == 'celltypist':
        return run_celltypist(loader_gen, checkpoint_path, cell_type_hierarchy_file)
    
def print_clf_report_per_class(clf_report_per_class, cell_type_mapping_file, title=None):
    """Creates a bar plot of F1 scores for each cell type"""
    
    # Load cell type names
    cell_type_mapping = pd.read_parquet(cell_type_mapping_file)
    
    # Set wide figure size for better readability
    plt.rcParams['figure.figsize'] = (20, 3)
    
    # Create bar plot with cell type names
    ax = sns.barplot(
        data=(
            clf_report_per_class
            .assign(
                # Convert numeric indices back to cell type names
                cell_type=lambda df: df.index.to_series().astype(int).replace(cell_type_mapping['label'].to_dict())
            )
            .sort_values('f1-score', ascending=False)  # Sort by performance
        ),
        x='cell_type',
        y='f1-score',
        color='#1f77b4'
    )
    
    # Adjust plot formatting
    ax.tick_params(axis='x', labelrotation=90)  # Rotate labels for readability
    ax.set_xlabel('')
    ax.set_title(title)
    ax.xaxis.set_tick_params(labelsize='small');