# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from gafl.data.pdb_dataloader import LengthBatcher, get_num_batches
from gafl.data import residue_constants

from backflip.data import utils as du

PICKLE_EXTENSIONS = ['.pkl', '.pickle', '.pck', '.db', '.pck']

_MUT_RE = re.compile(r'^(?:(?P<chain>[A-Za-z0-9]):)?(?P<wt>[A-Z])(?P<pos>\d+)(?P<mut>[A-Z])$')


def _parse_mutations(mutation_str):
    if mutation_str is None or (isinstance(mutation_str, float) and np.isnan(mutation_str)):
        return []
    if isinstance(mutation_str, list):
        muts = mutation_str
    else:
        muts = re.split(r'[;,]\s*', str(mutation_str).strip())
    parsed = []
    for mut in muts:
        if not mut:
            continue
        match = _MUT_RE.match(mut)
        if match is None:
            raise ValueError(f'Invalid mutation format: {mut}')
        parsed.append({
            'chain': match.group('chain'),
            'wt': match.group('wt'),
            'pos': int(match.group('pos')),
            'mut': match.group('mut'),
        })
    return parsed


def _load_manifest(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() in ['.jsonl', '.json']:
        with open(path, 'r') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return pd.DataFrame(rows)
    raise ValueError(f'Unsupported manifest format: {path}')


def _restype_to_idx(restype):
    if restype in residue_constants.restype_order:
        return residue_constants.restype_order[restype]
    raise ValueError(f'Unknown residue type: {restype}')


def _infer_num_res(processed_path):
    path_extension = Path(processed_path).suffix
    if path_extension in PICKLE_EXTENSIONS:
        processed_feats = du.read_pkl(processed_path)
        processed_feats = du.parse_chain_feats(processed_feats)
        modeled_idx = processed_feats['modeled_idx']
    elif path_extension == '.pdb' or path_extension == '.gz':
        processed_feats = du.parse_pdb_feats(
            pdb_name=Path(processed_path).stem,
            pdb_path=processed_path,
            chain_id='A',
        )
        modeled_idx = processed_feats['residue_index']
    elif path_extension == '.npz':
        processed_feats, _ = du.read_npz(processed_path, conf_idx=None)
        processed_feats = du.parse_npz_feats(npz_feats=processed_feats)
        modeled_idx = processed_feats['residue_index']
    else:
        raise ValueError(f'Unknown file extension {path_extension}')
    if len(modeled_idx) == 0:
        raise ValueError(f'No modeled residues found in {processed_path}')
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    return int(max_idx - min_idx + 1)


def _pad_1d(values, pad_value=-1, dtype=torch.long):
    max_len = max(v.shape[0] for v in values)
    out = torch.full((len(values), max_len), pad_value, dtype=dtype)
    mask = torch.zeros((len(values), max_len), dtype=torch.float32)
    for i, v in enumerate(values):
        if v.numel() == 0:
            continue
        out[i, :v.shape[0]] = v
        mask[i, :v.shape[0]] = 1.0
    return out, mask


def ddg_collate_fn(batch):
    stacked = {}
    tensor_keys = [
        'aatype', 'res_idx', 'rotmats_1', 'trans_1', 'res_mask',
        'mut_mask', 'ddg'
    ]
    for key in tensor_keys:
        stacked[key] = torch.stack([b[key] for b in batch], dim=0)

    mut_pos, mut_pos_mask = _pad_1d([b['mut_pos'] for b in batch])
    wt_aa, _ = _pad_1d([b['wt_aa'] for b in batch])
    mut_aa, _ = _pad_1d([b['mut_aa'] for b in batch])
    stacked['mut_pos'] = mut_pos
    stacked['mut_pos_mask'] = mut_pos_mask
    stacked['wt_aa'] = wt_aa
    stacked['mut_aa'] = mut_aa
    stacked['meta'] = [b['meta'] for b in batch]
    return stacked


class DDGDataset(Dataset):
    def __init__(self, *, dataset_cfg, split):
        self._log = logging.getLogger(__name__)
        self._dataset_cfg = dataset_cfg
        self._split = split

        OmegaConf.set_struct(self._dataset_cfg, False)
        if not hasattr(self._dataset_cfg, 'pick_random_conf'):
            self._dataset_cfg.pick_random_conf = False
        if not hasattr(self._dataset_cfg, 'num_confs'):
            self._dataset_cfg.num_confs = None
        if not hasattr(self._dataset_cfg, 'pick_random_conf_prob'):
            self._dataset_cfg.pick_random_conf_prob = 0.2
        if not hasattr(self._dataset_cfg, 'dev_run'):
            self._dataset_cfg.dev_run = False
        if not hasattr(self._dataset_cfg, 'dev_num_examples'):
            self._dataset_cfg.dev_num_examples = 10
        if not hasattr(self._dataset_cfg, 'max_rows'):
            self._dataset_cfg.max_rows = None
        if not hasattr(self._dataset_cfg, 'seed'):
            self._dataset_cfg.seed = 123
        OmegaConf.set_struct(self._dataset_cfg, True)

        self.csv = _load_manifest(self._dataset_cfg.manifest_path)
        if split is not None and 'split' in self.csv.columns:
            self.csv = self.csv[self.csv['split'] == split].reset_index(drop=True)

        if self._dataset_cfg.dev_run:
            self.csv = self.csv.head(int(self._dataset_cfg.dev_num_examples)).reset_index(drop=True)
        if self._dataset_cfg.max_rows is not None:
            self.csv = self.csv.head(int(self._dataset_cfg.max_rows)).reset_index(drop=True)

        if 'modeled_seq_len' not in self.csv.columns:
            self._log.info('Inferring modeled_seq_len for DDG manifest.')
            path_col = 'processed_path' if 'processed_path' in self.csv.columns else 'pdb_path'
            self.csv['modeled_seq_len'] = [
                _infer_num_res(path) for path in self.csv[path_col]
            ]

    def __len__(self):
        return len(self.csv)

    def _process_structure(self, processed_file_path, pick_random_conf=False, num_confs=None, rng=None, chain_id=None):
        path_extension = Path(processed_file_path).suffix
        pkl_file = False
        pdb_file = False
        if path_extension in PICKLE_EXTENSIONS:
            if pick_random_conf:
                pick_random_conf = False
            pkl_file = True
            processed_feats = du.read_pkl(processed_file_path)
            processed_feats = du.parse_chain_feats(processed_feats)
            modeled_idx = processed_feats['modeled_idx']
            feat_dict = {}
        elif path_extension == '.pdb' or path_extension == '.gz':
            pdb_file = True
            processed_feats = du.parse_pdb_feats(
                pdb_name=Path(processed_file_path).stem,
                pdb_path=processed_file_path,
                chain_id=chain_id or 'A',
            )
            modeled_idx = processed_feats['residue_index']
            feat_dict = {}
        elif path_extension == '.npz':
            conf_idx = None
            if pick_random_conf:
                if num_confs is None:
                    raise ValueError('num_confs must be provided if pick_random_conf is True')
                if rng is None:
                    rng = np.random.default_rng(self._dataset_cfg.seed)
                if rng.random() < self._dataset_cfg.pick_random_conf_prob:
                    conf_idx = int(rng.integers(0, num_confs))
            processed_feats, feat_dict = du.read_npz(processed_file_path, conf_idx=conf_idx)
            processed_feats = du.parse_npz_feats(npz_feats=processed_feats)
            modeled_idx = processed_feats['residue_index']
        else:
            raise ValueError(f'Unknown file extension {path_extension}')

        if len(modeled_idx) == 0:
            raise ValueError(f'No modeled residues found in {processed_file_path}')

        if not pkl_file:
            modeled_idx = modeled_idx - 1
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        processed_feats = {
            k: v[min_idx:(max_idx + 1)] for k, v in processed_feats.items()
        }
        processed_feats['pdb_name'] = processed_file_path.split('/')[-1].split('.')[0]

        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).float(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).float(),
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        if pkl_file:
            res_idx_raw = processed_feats['modeled_idx']
        else:
            res_idx_raw = processed_feats['residue_index']
        res_idx = res_idx_raw - np.min(res_idx_raw) + 1

        return {
            'aatype': chain_feats['aatype'],
            'res_idx': res_idx,
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,
            'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
            'pdb_name': processed_feats['pdb_name'],
            'res_idx_raw': res_idx_raw,
        }

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        processed_file_path = row['processed_path'] if 'processed_path' in row else row['pdb_path']
        chain_id = row['chain_id'] if 'chain_id' in row else None
        if chain_id is not None and (isinstance(chain_id, float) and np.isnan(chain_id) or chain_id == ""):
            chain_id = None
        rng = np.random.default_rng(self._dataset_cfg.seed + int(idx))
        chain_feats = self._process_structure(
            processed_file_path=processed_file_path,
            pick_random_conf=self._dataset_cfg.pick_random_conf,
            num_confs=self._dataset_cfg.num_confs,
            rng=rng,
            chain_id=chain_id,
        )

        mutations = _parse_mutations(row['mutation'])
        res_idx_raw = np.array(chain_feats['res_idx_raw'])
        res_idx_map = {int(r): i + 1 for i, r in enumerate(res_idx_raw)}
        num_res = chain_feats['res_mask'].shape[0]

        mut_positions = []
        wt_aa = []
        mut_aa = []
        for mut in mutations:
            pos = mut['pos']
            if pos in res_idx_map:
                pos = res_idx_map[pos]
            if pos < 1 or pos > num_res:
                raise ValueError(f'Mutation position {mut["pos"]} out of range for {processed_file_path}')
            mut_positions.append(pos)
            wt_aa.append(_restype_to_idx(mut['wt']))
            mut_aa.append(_restype_to_idx(mut['mut']))

            ref_aatype = int(chain_feats['aatype'][pos - 1])
            if ref_aatype != wt_aa[-1]:
                self._log.warning(
                    f'WT AA mismatch at {processed_file_path} pos {mut["pos"]}: '
                    f'manifest {mut["wt"]} vs aatype {ref_aatype}'
                )

        mut_mask = torch.zeros(num_res, dtype=torch.float32)
        for pos in mut_positions:
            mut_mask[pos - 1] = 1.0

        ddg = torch.tensor(float(row['ddg']), dtype=torch.float32)
        mut_pos = torch.tensor([p - 1 for p in mut_positions], dtype=torch.long)

        return {
            'aatype': chain_feats['aatype'],
            'res_idx': torch.tensor(chain_feats['res_idx']).long(),
            'rotmats_1': chain_feats['rotmats_1'],
            'trans_1': chain_feats['trans_1'],
            'res_mask': chain_feats['res_mask'].float(),
            'mut_mask': mut_mask,
            'mut_pos': mut_pos,
            'wt_aa': torch.tensor(wt_aa, dtype=torch.long),
            'mut_aa': torch.tensor(mut_aa, dtype=torch.long),
            'ddg': ddg,
            'meta': {
                'protein_id': row.get('protein_id', chain_feats['pdb_name']),
                'mutation': row['mutation'],
            }
        }


class DDGDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
        self.sampler_cfg = data_cfg.sampler

    def setup(self, stage: str):
        assert self.dataset_cfg.seed is not None, 'seed must be provided in the dataset config'

        train_cfg = OmegaConf.create(OmegaConf.to_container(self.dataset_cfg, resolve=True))
        val_cfg = OmegaConf.create(OmegaConf.to_container(self.dataset_cfg, resolve=True))
        test_cfg = OmegaConf.create(OmegaConf.to_container(self.dataset_cfg, resolve=True))

        use_split_manifests = (
            hasattr(self.dataset_cfg, 'train_manifest') and
            self.dataset_cfg.train_manifest is not None
        )
        if use_split_manifests:
            train_cfg.manifest_path = self.dataset_cfg.train_manifest
            val_cfg.manifest_path = self.dataset_cfg.val_manifest
            test_cfg.manifest_path = self.dataset_cfg.test_manifest
            train_split = None
            val_split = None
            test_split = None
        else:
            if not hasattr(self.dataset_cfg, 'manifest_path') or self.dataset_cfg.manifest_path is None:
                raise ValueError("manifest_path must be set when train/val/test manifests are not provided.")
            train_cfg.manifest_path = self.dataset_cfg.manifest_path
            val_cfg.manifest_path = self.dataset_cfg.manifest_path
            test_cfg.manifest_path = self.dataset_cfg.manifest_path
            train_split = 'train'
            val_split = 'val'
            test_split = 'test'

        train_cfg.pick_random_conf = False
        val_cfg.pick_random_conf = False
        test_cfg.pick_random_conf = False
        if hasattr(self.dataset_cfg, 'train_max_rows'):
            train_cfg.max_rows = self.dataset_cfg.train_max_rows
        if hasattr(self.dataset_cfg, 'val_max_rows'):
            val_cfg.max_rows = self.dataset_cfg.val_max_rows
        if hasattr(self.dataset_cfg, 'test_max_rows'):
            test_cfg.max_rows = self.dataset_cfg.test_max_rows

        self._train_cfg = train_cfg
        self._val_cfg = val_cfg
        self._test_cfg = test_cfg
        self._train_split = train_split
        self._val_split = val_split
        self._test_split = test_split

        self._train_dataset = DDGDataset(dataset_cfg=self._train_cfg, split=self._train_split)
        self._valid_dataset = DDGDataset(dataset_cfg=self._val_cfg, split=self._val_split)
        self._test_dataset = DDGDataset(dataset_cfg=self._test_cfg, split=self._test_split)

        self._validate_manifests(use_split_manifests)

        logging.info(f'Train dataset: {len(self._train_dataset)} examples')
        logging.info(f'Valid dataset: {len(self._valid_dataset)} examples')
        logging.info(f'Test dataset: {len(self._test_dataset)} examples')

    def train_dataloader(self):
        return make_ddg_dataloader(
            dataset_cfg=self._train_cfg,
            split=self._train_split,
            batch_size=self.loader_cfg.batch_size,
            num_workers=self.loader_cfg.num_workers,
            seed=self.dataset_cfg.seed,
            sampler_cfg=self.sampler_cfg,
            is_train=True,
        )

    def val_dataloader(self):
        return [
            make_ddg_dataloader(
                dataset_cfg=self._val_cfg,
                split=self._val_split,
                batch_size=self.loader_cfg.batch_size,
                num_workers=self.loader_cfg.num_workers,
                seed=self.dataset_cfg.seed,
                sampler_cfg=self.sampler_cfg,
            )
        ]

    def test_dataloader(self):
        return [
            make_ddg_dataloader(
                dataset_cfg=self._test_cfg,
                split=self._test_split,
                batch_size=self.loader_cfg.batch_size,
                num_workers=self.loader_cfg.num_workers,
                seed=self.dataset_cfg.seed,
                sampler_cfg=self.sampler_cfg,
            )
        ]

    def _validate_manifests(self, use_split_manifests):
        if use_split_manifests:
            for split, cfg in [('train', self._train_cfg), ('val', self._val_cfg), ('test', self._test_cfg)]:
                path = Path(cfg.manifest_path)
                if not path.exists():
                    raise FileNotFoundError(f"Missing manifest for {split}: {path}")
                df = _load_manifest(path)
                if 'split' in df.columns:
                    bad = df[df['split'] != split]
                    if len(bad) > 0:
                        raise ValueError(f"Manifest {path} has split values not equal to '{split}'.")
        else:
            path = Path(self._train_cfg.manifest_path)
            if not path.exists():
                raise FileNotFoundError(f"Missing manifest: {path}")
            df = _load_manifest(path)
            if 'split' not in df.columns:
                raise ValueError(f"Manifest {path} must include a split column.")
            allowed = {'train', 'val', 'test'}
            bad = df[~df['split'].isin(allowed)]
            if len(bad) > 0:
                raise ValueError(f"Manifest {path} has invalid split labels.")

def make_ddg_dataloader(*, dataset_cfg, split, batch_size, num_workers, seed, sampler_cfg=None, is_train=False):
    dataset = DDGDataset(dataset_cfg=dataset_cfg, split=split)
    if sampler_cfg is None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train or (split == 'train'),
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=ddg_collate_fn,
        )

    batch_sampler = LengthBatcher(
        sampler_cfg=sampler_cfg,
        metadata_csv=dataset.csv,
        seed=seed,
        shuffle=is_train or (split == 'train'),
        rank=None,
        num_replicas=None,
        num_batches=get_num_batches(sampler_cfg, dataset.csv, seed=seed),
    )
    prefetch_factor = None if num_workers == 0 else 2
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=ddg_collate_fn,
    )
