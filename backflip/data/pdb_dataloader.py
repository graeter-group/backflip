# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PDB data loader."""
import math
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
from gafl.data.pdb_dataloader import LengthBatcher, get_num_batches
from gafl.data.pdb_dataloader import EmptyDataset
from gafl.data.pdb_dataloader import PdbDataset as PdbDatasetGAFL

from backflip.data import utils as du

PICKLE_EXTENSIONS = ['.pkl', '.pickle', '.pck', '.db', '.pck']


class PdbDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset

        # # Handle missing generate_valid_samples argument
        # if not hasattr(self.dataset_cfg, 'generate_valid_samples'):
        #     self.dataset_cfg.generate_valid_samples = False
        # OmegaConf.set_struct(self.dataset_cfg, True)
            
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

class PdbDataset(PdbDatasetGAFL):
    def __init__(
            self,
            *,
            dataset_cfg,
        ):
        self.flexibility = dataset_cfg.flexibility

        # set some gafl-data values that we dont want to change for flexibility prediction
        OmegaConf.set_struct(dataset_cfg, False)
        dataset_cfg.filter_breaks = False
        dataset_cfg.label_breaks = False
        dataset_cfg.use_res_idx = False
        dataset_cfg.filter_scrmsd = "inf"
        dataset_cfg.max_coil_pct = 1.
        # set the split for the dataset to None, we split before creating the dataset. (its called partition in the dataset config):
        dataset_cfg.train_valid_test_split = [1.0, 0.0, 0.0]
        dataset_cfg.calc_dssp = False
        OmegaConf.set_struct(dataset_cfg, True)


        # is_training is gafl-specific for splitting, here we set it to True also for val and test!
        return super().__init__(dataset_cfg=dataset_cfg, is_training=True)


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

        # # check whether there are duplicate idxs:
        # if len(set(modeled_idx)) != len(modeled_idx):
        #     raise ValueError(f'Duplicate residue indices found in {processed_file_path}')

        # Filter out residues that are not modeled.
        # assuming starting from 1 [0:len+1] then
        if pkl_file == False:
            modeled_idx -= 1
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        processed_feats = tree.map_structure(
                lambda x: x[min_idx:(max_idx+1)], processed_feats)
        processed_feats['pdb_name'] = processed_file_path.split('/')[-1].split('.')[0]

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
            if feat_name in feat_dict.keys():
                extra_feats[feat_name] = torch.tensor(feat_dict[feat_name]).float()
                # assert that the zeroth and first dimensions fit:
                num_res = trans_1.shape[0]
                assert extra_feats[feat_name].shape[0] == num_res, f'{feat_name} shape[0] {extra_feats[feat_name].shape[0]} != {num_res}'
                if len(extra_feats[feat_name].shape) == 1:
                    extra_feats[feat_name] = extra_feats[feat_name].unsqueeze(-1) 
            else:
                raise ValueError(f'Feature {feat_name} not found in npz file')


        d = {
            'aatype': chain_feats['aatype'],
            'res_idx': res_idx,
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,
            'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
            'pdb_name': processed_feats['pdb_name'],
        }
        d.update(extra_feats)
        # for k, v in d.items():
        #     if hasattr(v, 'shape'):
        #         print(f'{k}: {v.shape}')
        # exit()
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
