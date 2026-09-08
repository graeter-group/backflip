# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PDB data loader."""
import math
import os
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging
from tqdm import tqdm
from copy import deepcopy

from openfold.data import data_transforms
from openfold.utils import rigid_utils

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist
from omegaconf import OmegaConf
from pathlib import Path

from backflip.data import utils as du

PICKLE_EXTENSIONS = ['.pkl', '.pickle', '.pck', '.db', '.pck']


class LengthBatcher:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
            num_batches=None,
        ):
        """
        If clustered: Assumes that there is an entry "cluster" in the metadata, for which a unique identifier is expected. Then, an epoch is defined as iteration over all clusters, where for each epoch, one random cluster member is picked.
        """
        # In FrameFlow, each epoch had n_data batches, so the data points were seen multiple times per epoch.
        # here, we define epoch as the iteration over all data points once.
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        self._cluster_init()
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

        # OUTCOMMENTED BECAUSE DDP REQUIRES SAME NUM BATCHES ACROSS ALL REPLICAS!
        ###############################
        # # set max num batches to be the number of batches in epoch 0.
        # self.num_batches = None
        # self._create_batches()
        # self.num_batches = len(self.sample_order)
        ###############################


        self.num_batches = num_batches
        if num_batches is not None:
            self.num_batches = num_batches//self.num_replicas
            self._log.info(f'Total number of batches: {self.num_batches}')
            self._log.info(f'Number of batches per replica: {self.num_batches}')

    def _cluster_init(self):
        self.clustered = self._sampler_cfg.clustered if hasattr(self._sampler_cfg, 'clustered') else False
        if self.clustered:
            if not 'cluster' in self._data_csv.columns:
                raise ValueError("Clustered sampling requires a 'cluster' column in the metadata.")

            def _cluster_unassigned(cluster_id):
                if cluster_id in ["", "None", "nan", None]:
                    return True
                if isinstance(cluster_id, float) or isinstance(cluster_id, int):
                    return np.isnan(cluster_id)
                return False

            # assign clusters to states that have no valid cluster entry in the metadata.csv:
            for i, row in self._data_csv.iterrows():
                if _cluster_unassigned(row['cluster']):
                    new_cluster = hash(str(i)) # assume that this does not occur twice
                    self._data_csv.at[i, 'cluster'] = new_cluster

                elif not isinstance(row['cluster'], str):
                    if float(row['cluster']).is_integer():
                        self._data_csv.at[i, 'cluster'] = int(row['cluster'])
                    else:
                        raise ValueError(f"Cluster must be a string or integer-like float. Found {row['cluster']} in {row['pdb_name']}.")


            all_clusters = self._data_csv['cluster'].unique()
            has_nan_cluster = any([cluster == "nan" for cluster in all_clusters])
            if has_nan_cluster and len(all_clusters) == 1:
                raise RuntimeError("Internal error. All clusters are nan.")
            if has_nan_cluster:
                logging.warning(f"Found nan cluster among {len(all_clusters)} clusters. This will be treated as a separate cluster.")

            # dictionary that maps cluster to a list of csv indices that belong to that cluster
            self.cluster_dict = {cluster: self._data_csv[self._data_csv['cluster'] == cluster].index.tolist() for cluster in all_clusters}
            assert sum([len(self.cluster_dict[cluster]) for cluster in self.cluster_dict]) == len(self._data_csv), "Internal error. Not all indices are assigned to a cluster."

            logging.info(f'Applying clustered sampling: {len(all_clusters)} clusters, {len(self._data_csv)} examples.')


    def _get_replica_csv(self, rng:torch.Generator):
        replica_csv = self._data_csv

        if self.clustered:
            np_rng = np.random.default_rng(seed=int(rng.initial_seed()))
            # for each cluster, choose a random member and write it to replica_csv
            cluster_indices = []
            for cluster in self.cluster_dict:
                cluster_indices.append(np_rng.choice(self.cluster_dict[cluster], size=1)[0])

            replica_csv = replica_csv[replica_csv.index.isin(cluster_indices)]

        # indices are only used for randomly splitting the batch across devices
        indices = list(range(len(replica_csv)))

        if self.shuffle:
            indices = torch.randperm(len(indices), generator=rng).tolist()

        # replica csv is a reordered version (and for num_replicas > 1 also subset of) the original csv
        if len(indices) > self.num_replicas:
            replica_csv = replica_csv.iloc[
                indices[self.rank::self.num_replicas]
            ]

        return replica_csv


    def _replica_epoch_batches(self):
        """
        Returns the batch idxs for the current epoch.
        """
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)

        replica_csv = self._get_replica_csv(rng)

        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                sample_order.append(batch_indices)

        # Remove any length bias.
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        return [sample_order[i] for i in new_order]

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = self._replica_epoch_batches()
        if self.num_batches is not None:
            if len(all_batches) > self.num_batches:
                all_batches = all_batches[:self.num_batches]
            elif len(all_batches) < self.num_batches:
                # randomly duplicate batches to match the number of batches
                duplicates = np.random.choice(len(all_batches), size=self.num_batches - len(all_batches), replace=True)
                all_batches.extend([all_batches[d] for d in duplicates])
        else:
            raise ValueError("num_batches must be set before creating batches")
        self.sample_order = all_batches


    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self.num_batches


def get_num_batches(sampler_cfg, metadata_csv, seed=123):
    """
    Function to infer the number of batches in an epoch. This can be used for automatically inferring the approx. (exact number is epoch-dependent) total number of batches for LengthBatcher. Has to be called before initializing the actual distributed samplers since the number of batches has to be the same across all replicas.
    """
    # create a dummy length batcher with num_replicas=1:
    batcher = LengthBatcher(
        sampler_cfg=sampler_cfg,
        metadata_csv=metadata_csv,
        seed=seed,
        shuffle=True,
        num_replicas=1,
        rank=0,
        num_batches=None
    )
    all_batches = batcher._replica_epoch_batches()
    return len(all_batches)


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty")


def _read_clusters_framediff(dataset_cfg):
    pdb_to_cluster = {}
    with open(dataset_cfg.cluster_path_framediff, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.lower()] = str(i)
    return pdb_to_cluster

def _get_pdb_to_cluster_dict_framediff(dataset_cfg):
    pdb_to_cluster = _read_clusters_framediff(dataset_cfg)
    return pdb_to_cluster


class PdbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
            
        self.sampler_cfg = data_cfg.sampler
        if not hasattr(self.sampler_cfg, 'clustered'):
            self.sampler_cfg.clustered = False
        if not hasattr(self.dataset_cfg, 'extra_features'):
            self.dataset_cfg.extra_features = []
        if not hasattr(self.dataset_cfg, 'pick_random_conf_prob'):
            self.dataset_cfg.pick_random_conf_prob = 0.2

    def setup(self, stage: str):

        assert self.dataset_cfg.seed is not None, 'seed must be provided in the dataset config'

        train_cfg = deepcopy(self.dataset_cfg)
        val_cfg = deepcopy(self.dataset_cfg)
        test_cfg = deepcopy(self.dataset_cfg)

        train_cfg.csv_path = self.dataset_cfg.train_csv_path
        val_cfg.csv_path = self.dataset_cfg.val_csv_path
        test_cfg.csv_path = self.dataset_cfg.test_csv_path

        val_cfg.pick_random_conf = False
        test_cfg.pick_random_conf = False
        
        self._train_dataset = PdbDataset(
            dataset_cfg=train_cfg,
        )
        self._valid_dataset = PdbDataset(
            dataset_cfg=val_cfg,
        )
        self._test_dataset = PdbDataset(
            dataset_cfg=test_cfg,
        )

        logging.info(f'Train dataset: {len(self._train_dataset)} examples')
        logging.info(f'Valid dataset: {len(self._valid_dataset)} examples')
        logging.info(f'Test dataset: {len(self._test_dataset)} examples')


    def train_dataloader(self, rank=None, num_replicas=None):
        if self.dataset_cfg.train_csv_path is None:
            train_loader = DataLoader(EmptyDataset(), batch_size=1)
        else:
            num_workers = self.loader_cfg.num_workers
            batch_sampler = LengthBatcher(
                    sampler_cfg=self.sampler_cfg,
                    metadata_csv=self._train_dataset.csv,
                    rank=rank,
                    num_replicas=num_replicas,
                    num_batches=get_num_batches(self.sampler_cfg, self._train_dataset.csv)
                )
            train_loader = DataLoader(
                self._train_dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
                pin_memory=False,
                persistent_workers=True if num_workers > 0 else False,
            )

        return train_loader

    def val_dataloader(self):
        if self.dataset_cfg.val_csv_path is None:
            valid_loader = DataLoader(EmptyDataset(), batch_size=1)
        else:
            num_workers = self.loader_cfg.num_workers
            valid_loader = DataLoader(
                                self._valid_dataset,
                                # changed to the batch_sampler
                                batch_sampler=LengthBatcher(
                                    sampler_cfg=self.sampler_cfg,
                                    metadata_csv=self._valid_dataset.csv,
                                    rank=None,
                                    num_replicas=None,
                                    num_batches=get_num_batches(self.sampler_cfg, self._valid_dataset.csv),
                                ),
                                num_workers=num_workers,
                                prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
                                pin_memory=False,
                                persistent_workers=True if num_workers > 0 else False,
                            )

        return [valid_loader]

    def test_dataloader(self):
        if self.dataset_cfg.test_csv_path is None:
            test_loader = DataLoader(EmptyDataset(), batch_size=1)
        else:
            num_workers = self.loader_cfg.num_workers
            test_loader = DataLoader(
                                self._test_dataset,
                                # changed to the batch_sampler
                                batch_sampler=LengthBatcher(
                                    sampler_cfg=self.sampler_cfg,
                                    metadata_csv=self._test_dataset.csv,
                                    rank=None,
                                    num_replicas=None,
                                    num_batches=get_num_batches(self.sampler_cfg, self._test_dataset.csv),
                                ),
                                num_workers=num_workers,
                                prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
                                pin_memory=False,
                                persistent_workers=True if num_workers > 0 else False,
                            )

        return [test_loader]

class PdbDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
        ):
        self._log = logging.getLogger(__name__)
        # is_training controls splitting logic in _init_metadata; here we set it to True also for val and test,
        # since the actual train/val/test split is done upstream (by pointing csv_path at separate metadata files).
        self._is_training = True
        self.sample_dataset = False
        self.flexibility = dataset_cfg.flexibility

        # set some values that we dont want to change for flexibility prediction
        OmegaConf.set_struct(dataset_cfg, False)
        dataset_cfg.filter_breaks = False
        dataset_cfg.label_breaks = False
        dataset_cfg.use_res_idx = False
        dataset_cfg.filter_scrmsd = "inf"
        dataset_cfg.max_coil_pct = 1.
        # set the split for the dataset to None, we split before creating the dataset. (its called partition in the dataset config):
        dataset_cfg.train_valid_test_split = [1.0, 0.0, 0.0]
        dataset_cfg.calc_dssp = False

        if not hasattr(dataset_cfg, 'break_csv_path'):
            dataset_cfg.break_csv_path = dataset_cfg.csv_path.replace('metadata.csv', 'breaks.csv')
        if not hasattr(dataset_cfg, 'min_num_res_eval'):
            dataset_cfg.min_num_res_eval = 60
        if not hasattr(dataset_cfg, 'allowed_oligomers'):
            dataset_cfg.allowed_oligomers = None
        if not hasattr(dataset_cfg, 'apply_clustering'):
            dataset_cfg.apply_clustering = False
        if not hasattr(dataset_cfg, 'target_sec_content'):
            dataset_cfg.target_sec_content = OmegaConf.create()
            dataset_cfg.target_sec_content.helix_percent = 0.32
            dataset_cfg.target_sec_content.strand_percent = 0.27
        OmegaConf.set_struct(dataset_cfg, True)

        self._dataset_cfg = dataset_cfg
        self._init_metadata()
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        pdb_csv = pd.read_csv(self.dataset_cfg.csv_path)

        # apply naming convention
        pdb_csv = du.metadata_naming_convention(pdb_csv)

        if self.dataset_cfg.apply_clustering:
            if hasattr(self.dataset_cfg, 'cluster_path_framediff'):

                # assume that the dataset type is framediff and read the clusters from the cluster file
                logging.info("Reading and clusters from framdiff-like-cluster file and storing to metadata...\nIf you want to use the cluster present in the metadata, set cluster_path_framediff to null.")
                pdb_to_cluster = _get_pdb_to_cluster_dict_framediff(self.dataset_cfg)
                # store the cluster information in the metadata
                cluster = []
                for pdb_name in pdb_csv['pdb_name']:
                    cluster.append(pdb_to_cluster[pdb_name] if pdb_name in pdb_to_cluster else "nan")

                unique_clusters = set(list(cluster))
                logging.info(f"Found {len(unique_clusters)} unique clusters.\n")
                pdb_csv['cluster'] = cluster

        # define a filter mask that is used to filter the dataset
        ###############################
        filter_mask = [True] * len(pdb_csv)

        # filter for breaks (which we define as non-continuous residue indices)
        if self.dataset_cfg.filter_breaks:
            # Check if column "breaks" exists
            if 'dist_breaks' not in pdb_csv.columns:
                breaks = []
                for path in tqdm(pdb_csv['processed_path'], desc='Saving dist breaks in metadata...'):
                    chain_feats = self._process_csv_row(path)

                    breaks.append(du.has_breaks(chain_feats) or du.has_inconstistent_indexing(chain_feats))
                pdb_csv['dist_breaks'] = breaks
                pdb_csv.to_csv(self.dataset_cfg.csv_path, index=False)

            filter_mask_breaks = pdb_csv['dist_breaks'] == False
            logging.info(f'Filtering for breaks: Removed {len(pdb_csv) - sum(filter_mask_breaks)} of {len(pdb_csv)} examples. {sum(filter_mask_breaks)} remaining.')
            filter_mask = np.logical_and(filter_mask, filter_mask_breaks)

        # filter for scrmsd
        if self.dataset_cfg.filter_scrmsd not in ["inf", "nan", float('inf'), float('nan'), None]:
            max_scrmsd = float(self.dataset_cfg.filter_scrmsd)
            # Check if column "scrmsd" exists or whether a csv file with scrmsd values is provided, then write the scrmsd values to the csv.
            if 'scrmsd' not in pdb_csv.columns or hasattr(self.dataset_cfg, 'scrmsd_csv_path'):
                assert hasattr(self.dataset_cfg, 'scrmsd_csv_path'), 'scrmsd_csv_path must be provided in the config if scrmsd column is not present in the csv.'
                if not Path(self.dataset_cfg.scrmsd_csv_path).exists():
                    raise FileNotFoundError(f"File {self.dataset_cfg.scrmsd_csv_path} not found. Set filter_scrmsd to 'inf' to ignore this error.")
                scrmsd_csv = pd.read_csv(self.dataset_cfg.scrmsd_csv_path)
                scrmsd_dict = {scrmsd_csv['pdb'][i]: scrmsd_csv['scrmsd'][i] for i in range(len(scrmsd_csv))}
                scrmsd_values = []
                for pdb_name in pdb_csv['pdb_name']:
                    scrmsd_values.append(scrmsd_dict[pdb_name] if pdb_name in scrmsd_dict else "nan")
                pdb_csv['scrmsd'] = scrmsd_values
                pdb_csv.to_csv(self.dataset_cfg.csv_path, index=False)

            scrmsd_values = [float(v) if v != "nan" else float('inf') for v in pdb_csv['scrmsd']]
            filter_mask_scrmsd = (np.array(scrmsd_values) <= max_scrmsd).tolist()
            now_filtered_out = [filter_mask[i] and not filter_mask_scrmsd[i] for i in range(len(filter_mask))]
            currently_remaining = sum(filter_mask)
            filter_mask = np.logical_and(filter_mask, filter_mask_scrmsd)

            logging.info(f'Filtering for scrmsd < {max_scrmsd}. Removed {sum(now_filtered_out)} of {currently_remaining} examples. {sum(filter_mask)} remaining.')

        if self.dataset_cfg.max_coil_pct < 1.:
            # Check if column "coil_pct" exists
            assert 'coil_percent' in pdb_csv.columns, 'Column "coil_percent" must be present in the metadata csv. Cols: ' + str(pdb_csv.columns)
            filter_mask_coil = pdb_csv['coil_percent'] <= self.dataset_cfg.max_coil_pct

            now_filtered_out = [filter_mask[i] and not filter_mask_coil[i] for i in range(len(filter_mask))]
            currently_remaining = sum(filter_mask)
            filter_mask = np.logical_and(filter_mask, filter_mask_coil)
            logging.info(f'Filtering for coil_pct <= {self.dataset_cfg.max_coil_pct}. Removed {sum(now_filtered_out)} of {currently_remaining} examples. {sum(filter_mask)} remaining.')

        # oligomer filter (if not in metadata cols, assume all are monomeric)
        if self.dataset_cfg.allowed_oligomers is not None:
            if self.dataset_cfg.allowed_oligomers != ['monomeric'] and not 'oligomeric_detail' in pdb_csv.columns:
                raise ValueError('Column "oligomeric_detail" must be present in the metadata csv if allowed_oligomers is set.')
            allowed_oligomers = self.dataset_cfg.allowed_oligomers
            if 'oligomeric_detail' in pdb_csv.columns:
                filter_mask_oligomers = pdb_csv['oligomeric_detail'].isin(allowed_oligomers)
                now_filtered_out = [filter_mask[i] and not filter_mask_oligomers[i] for i in range(len(filter_mask))]
                currently_remaining = sum(filter_mask)
                filter_mask = np.logical_and(filter_mask, filter_mask_oligomers)
                logging.info(f'Filtering for allowed oligomers: Removed {sum(now_filtered_out)} of {currently_remaining} examples. {sum(filter_mask)} remaining.')
        ###############################


        # Process information of breaks (unmodelled residues) in the pdb files
        ###############################
        # be default, assume that the break path is metadata_path.parent/breaks.csv:
        if self.dataset_cfg.label_breaks or self.dataset_cfg.use_res_idx:
            if self.dataset_cfg.break_csv_path is None:
                self.dataset_cfg.break_csv_path = str(Path(self.dataset_cfg.csv_path).parent/'breaks.csv')
                logging.info(f'data.dataset.break_csv_path is None. Setting it to data.dataset.csv_path.parent/breaks.csv, i.e. {self.dataset_cfg.break_csv_path}')
            if os.path.exists(self.dataset_cfg.break_csv_path):
                logging.info(f'Loading break information from self.dataset_cfg.break_csv_path={self.dataset_cfg.break_csv_path}')
                self.break_csv = pd.read_csv(self.dataset_cfg.break_csv_path)
            else:
                logging.info('Found no break information at self.dataset_cfg.break_csv_path')
                logging.info(f'Calculating breaks for dataset and storing them at {self.dataset_cfg.break_csv_path}...')

                breaks_dict = {
                    'pdb_name': [],
                    'idx_breaks': [],
                    'dist_breaks': [],
                    'merged_idx': [],
                    'consistent_breaks': []
                }

                for i in tqdm(range(len(pdb_csv)), desc='Calculating positions of breaks'):
                    csv_row = pdb_csv.iloc[i]
                    data = self._process_csv_row(csv_row['processed_path'])
                    idx1 = du.idx_breaks(data)
                    idx2 = du.dist_breaks(data)
                    merged_idx = np.unique(np.concatenate((idx1, idx2)))
                    # True if IDX and DIST criteriums for breaks are met at the same time
                    consistent_breaks = np.array_equal(idx1, idx2)
                    breaks_dict['pdb_name'].append(csv_row['pdb_name'])
                    breaks_dict['idx_breaks'].append(list(idx1))
                    breaks_dict['dist_breaks'].append(list(idx2))
                    breaks_dict['merged_idx'].append(list(merged_idx))
                    breaks_dict['consistent_breaks'].append(consistent_breaks)

                self.break_csv = pd.DataFrame(breaks_dict)
                self.break_csv.to_csv(self.dataset_cfg.break_csv_path, index=False)

        ###############################


        # Note: pdb_csv is the stored csv file with all samples and most relevant columns, self.csv will be a) filtered and b) appended by additional columns like break information etc.
        self.csv = pdb_csv
        if self.dataset_cfg.use_res_idx or self.dataset_cfg.label_breaks:
            self.csv = self.csv.merge(self.break_csv, on='pdb_name')

        # apply filtering:
        if not isinstance(filter_mask, list):
            filter_mask = filter_mask.tolist()

        self.csv = self.csv[filter_mask]
        self.csv = self.csv.reset_index(drop=True)

        logging.info(f'Number of examples after filtering: {len(pdb_csv)}\n')
        # Filter for modeled sequence length.
        self.csv = self.csv[self.csv.modeled_seq_len <= self.dataset_cfg.max_num_res]
        self.csv = self.csv[self.csv.modeled_seq_len >= self.dataset_cfg.min_num_res]

        logging.info(f'Number of examples after filtering for modeled sequence length: {len(self.csv)}\n')

        if self.dataset_cfg.subset is not None:
            self.csv = self.csv.iloc[:self.dataset_cfg.subset]

        if self.dataset_cfg.calc_dssp:
            self.helix_percent = np.mean(self.csv['helix_percent'].to_numpy())
            self.strand_percent = np.mean(self.csv['strand_percent'].to_numpy())
        else:
            self.helix_percent = self.dataset_cfg.target_sec_content.helix_percent
            self.strand_percent = self.dataset_cfg.target_sec_content.strand_percent

        # Training or validation specific logic.
        if self.is_training:
            logging.info(f'Using {self.helix_percent} helix and {self.strand_percent} strand content as target secondary structure content')

            # Extract training set.
            if self.dataset_cfg.train_valid_test_split[0] != 1.0:
                self.csv = self.csv.groupby('modeled_seq_len')
                self.csv = self.csv.apply(lambda x: x.sample(frac=self.dataset_cfg.train_valid_test_split[0], replace=False, random_state=self.dataset_cfg.seed)).droplevel(0)
                self.csv = self.csv.sort_values('modeled_seq_len', ascending=False)
                self._log.info(
                    f'Training: {len(self.csv)} examples')
            else:
                self.csv = self.csv.sort_values('modeled_seq_len', ascending=False)
        else:
            if self.sample_dataset:
                mask = self.csv['modeled_seq_len'] >= self.dataset_cfg.min_num_res_eval
                mask &= self.csv['modeled_seq_len'] <= self.dataset_cfg.min_eval_length
                eval_csv = self.csv[mask]
                eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
                all_lengths = np.sort(eval_csv.modeled_seq_len.unique())
                length_indices = (len(all_lengths) - 1) * np.linspace(
                    0.0, 1.0, self.dataset_cfg.num_eval_lengths)
                length_indices = length_indices.astype(int)
                eval_lengths = all_lengths[length_indices]
                eval_csv = eval_csv[eval_csv.modeled_seq_len.isin(eval_lengths)]

                # Fix a random seed to get the same split each time.
                eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                    self.dataset_cfg.samples_per_eval_length, replace=True, random_state=123)
                eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
                self.csv = eval_csv
                if self.dataset_cfg.generate_valid_samples:
                    self._log.info(
                        f'Generate {len(self.csv)} validation samples with lengths {eval_lengths}')
            else:
                # Extract validation set.
                if self.dataset_cfg.train_valid_test_split[1] > 0:
                    valid_frac = self.dataset_cfg.train_valid_test_split[1] / (self.dataset_cfg.train_valid_test_split[1] + self.dataset_cfg.train_valid_test_split[2])
                else:
                    self.csv = None
                    return

                train_csv = self.csv.groupby('modeled_seq_len')
                self.train_csv = train_csv.apply(lambda x: x.sample(frac=self.dataset_cfg.train_valid_test_split[0], replace=False, random_state=self.dataset_cfg.seed)).droplevel(0)
                self.csv = self.csv.drop(train_csv.index)
                self.csv = self.csv.groupby('modeled_seq_len')
                self.csv = self.csv.apply(lambda x: x.sample(frac=valid_frac, replace=False, random_state=self.dataset_cfg.seed)).droplevel(0)
                self.csv = self.csv.sort_values('modeled_seq_len', ascending=False)
                self._log.info(f'Validation: {len(self.csv)} examples')

    def __len__(self):
        return len(self.csv)

    def _process_csv_row(self, processed_file_path, pick_random_conf:bool=False, num_confs:int=None):
        path_extension = Path(processed_file_path).suffix
        pkl_file = False
        if path_extension in PICKLE_EXTENSIONS:
            assert pick_random_conf == False, 'pick_random_conf is not implemented for pkl files'
            pkl_file = True
            processed_feats = du.read_pkl(processed_file_path)
            processed_feats = du.parse_chain_feats(processed_feats)
            modeled_idx = processed_feats['modeled_idx']

        elif path_extension == '.npz':
            if pick_random_conf:
                if num_confs is None:
                    raise ValueError('num_confs must be provided if pick_random_conf is True')
                if np.random.rand() < self._dataset_cfg.pick_random_conf_prob:
                    conf_idx = np.random.randint(0, num_confs)
                else:
                    conf_idx = None
            else:
                conf_idx = None
            # modify to load more generic feats and give arg in dataloader for this!
            processed_feats, feat_dict = du.read_npz(processed_file_path, conf_idx=conf_idx)
            processed_feats = du.parse_npz_feats(npz_feats=processed_feats)
            modeled_idx = processed_feats['residue_index']
            # here the actual residue indices which are modeled are stored in the residue_index field
        else:
            raise ValueError(f'Unknown file extension {path_extension}')
        
        if len(modeled_idx) == 0:
            raise ValueError(f'No modeled residues found in {processed_file_path}')

        # Filter out residues that are not modeled.
        # assuming starting from 1 [0:len+1] then
        # NOTE: not robust for mdCATH
        if pkl_file == False:
            modeled_idx -= 1
        
        # NOTE: why do we keep it here if we anyways assume that residue_index is non-breaky for atlas and mdcath? has to be reworked if we move to smth else
        # NOTE: empirically found it does not make a difference
        # min_idx = np.min(modeled_idx)
        # max_idx = np.max(modeled_idx)
        # processed_feats = tree.map_structure(
        #         lambda x: x[min_idx:(max_idx+1)], processed_feats)
        
        processed_feats['pdb_name'] = processed_file_path.split('/')[-1].split('.')[0]
        # NOTE: has to be done for mdCATH or to filter out all non canonicals
        if np.any(processed_feats['aatype'] >= 20):
            # let's fallback to alanine 0 if there's something non-canonical...
            processed_feats['aatype'] = np.where(processed_feats['aatype'] >= 20, 0, processed_feats['aatype'])

        # Run through OpenFold data transforms.
        # changed .double() to .float() to match the floating point of GAFL
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).float(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).float()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        res_idx = processed_feats['modeled_idx'] if pkl_file else processed_feats['residue_index']
        res_idx = res_idx - np.min(res_idx) + 1
        
        extra_feats = {}
        for feat_name in self._dataset_cfg.extra_features:
            if feat_name not in feat_dict:
                raise ValueError(f'Feature {feat_name} not found in npz file')

            num_res = trans_1.shape[0]
            x = feat_dict[feat_name]
            if not torch.is_tensor(x):
                x = torch.tensor(x)
                x = x.float() if x.dtype in [torch.float64, torch.float32] else x.long()

            if feat_name == 'esm_emb':
                # Accept (1, N, D) or (N, D)
                if x.ndim == 3:
                    assert x.shape[0] == 1, f'{feat_name} batch dim {x.shape[0]} != 1'
                    x = x.squeeze(0)
                assert x.ndim == 2, f'{feat_name} ndim {x.ndim} != 2'

                # Remove BOS/EOS if present
                if x.shape[0] == num_res + 2:
                    x = x[1:-1, :]
                assert x.shape[0] == num_res, f'{feat_name} shape[0] {x.shape[0]} != {num_res}'

            elif feat_name == 'esmfold_s_z':
                assert x.shape == (num_res, num_res, 128), \
                    f'{feat_name} shape {tuple(x.shape)} != ({num_res}, {num_res}, 128)'

            else:
                assert x.shape[0] == num_res, f'{feat_name} shape[0] {x.shape[0]} != {num_res}'

            if x.ndim == 1:
                x = x.unsqueeze(-1)
            extra_feats[feat_name] = x

        d = {
            'aatype': chain_feats['aatype'],
            'res_idx': res_idx,
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,
            'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
            'pdb_name': processed_feats['pdb_name'],
        }
        d.update(extra_feats)
        return d

    def __getitem__(self, idx):
        '''
        Args:
            pick_random_conf: bool, whether to pick a random conformation from the npz file coords.shape = [100, N, 3, 3], where 100=N_confs by default
        '''
        # Sample data example.
        example_idx = idx
        if isinstance(example_idx, list):
            example_idx = example_idx[0]

        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row['processed_path']
        # modified flexibility_base.yaml dataset to have flexibility as a parameter

        chain_feats = self._process_csv_row(processed_file_path=processed_file_path, pick_random_conf=self._dataset_cfg.pick_random_conf, num_confs=self._dataset_cfg.num_confs)
        chain_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        return chain_feats
