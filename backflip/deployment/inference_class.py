# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

from typing import List, Union
import torch
import numpy as np
from pathlib import Path
import os
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import logging
import warnings
import time

from backflip.models.flexibility_module import FlexibilityModule
from backflip.deployment.utils import ckpt_path_from_tag, estimate_max_batchsize, read_path, frames_from_pdb, save_prediction, parse_input_paths

class BackFlip:
    """Thin inference wrapper around a trained BackFlip checkpoint."""

    def __init__(self, ckpt_path: Union[str, Path], device: str="cuda", features: List[str]=None, confidence_intervals: bool=False, progress_bar: bool=True, rmsf_type: str='global_rmsf'):
        """
        Load a checkpoint and its training config for inference.

        The checkpoint directory must include the original config.yaml so that
        the model architecture (and aatype usage) matches training.
        """

        ckpt_dir = os.path.dirname(ckpt_path)
        config_path = os.path.join(ckpt_dir, 'config.yaml')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {os.path.join(ckpt_dir, 'config.yaml')} does not exist.")
        
        self._cfg = OmegaConf.load(config_path)
        self._cfg.experiment.checkpointer.dirpath = './'
        
        self.confidence_intervals = confidence_intervals
        self.features = features # NOTE: IMPLEMENT THAT ONLY SELECTED FEATS ARE OUTPUTTED
        assert features is None, 'This option is not implemented yet.'

        self.ckpt_path = ckpt_path
        self.device = device

        self.model = self.load_model(ckpt_path, device=device)
        self.model.to(device)
        
        self.use_aatype = (hasattr(self._cfg.model, 'node_features') and
                hasattr(self._cfg.model.node_features, 'embed_aatype') and
                self._cfg.model.node_features.embed_aatype)
        logging.info(f'[BackFlip] use_aatype: {self.use_aatype}')

        self.rmsf_type = rmsf_type
        assert self.rmsf_type in ['global_rmsf', 'local_flex'], f"rmsf_type must be 'global_rmsf' or 'local_flex'. Got {self.rmsf_type}."

        self.progress_bar = progress_bar

    @classmethod
    def from_tag(cls, tag: str="latest", device: str="cuda", features: List[str]=None, confidence_intervals: bool=False, progress_bar: bool=True):
        """
        Resolve a known tag to a local checkpoint (downloading if needed).

        The tag maps to a directory under models/ and may trigger a download if
        weights are not present locally.
        """
        ckpt_path = ckpt_path_from_tag(tag)
        return cls(ckpt_path, device, features, confidence_intervals, progress_bar)

        
    def to(self, device: str):
        """Move the loaded model and future inputs to a different device."""
        self.model.to(device)
        self.device = device

    def load_model(self, ckpt_path: Union[str, Path], device:str='cuda'):
        """
        Load model weights from a Lightning checkpoint into the bare model.

        The checkpoint stores weights under the "model." prefix; this method
        strips that prefix before loading into the FlexibilityModel instance.
        """
        ckpt_path = Path(ckpt_path)
        assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
        ckpt = torch.load(ckpt_path, map_location=device if device is not None else self.device, weights_only=False)
        model_ckpt = ckpt["state_dict"]
        model_ckpt = {k.replace('model.', ''): v for k, v in model_ckpt.items()}

        module = FlexibilityModule(self._cfg)
        model = module.model
        model.load_state_dict(model_ckpt)
        return model

    def __call__(self, batch: dict):
        """
        Run the model on a prepared batch dict.

        If res_mask is missing, a full mask is injected so that every residue
        contributes to the prediction.
        """
        if 'res_mask' not in batch.keys():
            batch['res_mask'] = torch.ones_like(batch['trans_1'][..., 0]).float()  # Default mask if not provided
        return self.model(batch)

    def predict_from_pdb(self, pdb_path: Union[str, Path, List[Path]], res_idx= None, batch_size:int=None, cuda_memory_GB:int=8)->Union[List[dict], dict]:
        """
        Predict flexibility profiles from one or more PDB files.

        This uses the backbone frame representation extracted from each PDB and
        forwards through the model. Inputs of equal length are batched together
        in `predict_from_frames` to avoid padding.

        Args:
            pdb_path (Union[str, Path, List[Path]]): Path to a PDB file or a list of paths to PDB files.
            res_idx (List[torch.Tensor], optional): Optional residue indices. If None, uses torch.arange per protein.
            batch_size (int, optional): Max batch size for same-length proteins. If None, estimate from cuda_memory_GB.
            cuda_memory_GB (int, optional): Used to estimate batch size when batch_size is None.
        Returns:
            Union[List[dict], dict]: List of dictionaries containing the model outputs for each PDB file, or a single dictionary if a single PDB file is provided. These are typically dictionaries with keys local_flex and global_rmsf.
        """
        list_passed = isinstance(pdb_path, list)
        if isinstance(pdb_path, (str, Path)):
            pdb_path = [pdb_path]
        assert isinstance(pdb_path, list), f'pdb_path must be a string, Path or a list of Paths. Got {type(pdb_path)}.'

        translations, rotations, aatypes = [], [], []
        for path in pdb_path:
            model_input = frames_from_pdb(pdb_path=path)
            translations.append(model_input['trans_1'])
            rotations.append(model_input['rotmats_1'])
            if self.use_aatype:
                assert "aatype" in model_input, f"'aatype' required by model but missing in {path}"
                aatypes.append(model_input["aatype"])

        prediction = self.predict_from_frames(translations=translations, rotations=rotations,cuda_memory_GB=8, res_idx=res_idx, batch_size=batch_size, stop_grad=True, aatypes=aatypes if self.use_aatype else None)

        if not list_passed:
            assert len(prediction) == 1, f'Internal Error: Expected a single prediction for a single PDB file, but got {len(prediction)} predictions.'
            return prediction[0]
        else:
            assert len(prediction) == len(pdb_path), f'Internal Error: Expected the number of predictions to match the number of input PDB files, but got {len(prediction)} predictions for {len(pdb_path)} input files.'
            return prediction

    def predict_from_frames(self, translations: List[torch.Tensor], rotations: List[torch.Tensor], res_idx=None, batch_size:int=None, cuda_memory_GB:int=8, stop_grad:bool=True, aatypes:List[torch.Tensor]=None)-> List[dict]:
        """
        Predict a set of flexibilities given translations and rotations. Batches proteins of same lengths together and then reorders the output such that the output list has the same order as the input list.

        Args:
            translations (List[torch.Tensor]): Per-protein translations, shape (L, 3).
            rotations (List[torch.Tensor]): Per-protein rotation matrices, shape (L, 3, 3).
            res_idx (List[torch.Tensor], optional): Residue indices per protein. If None, uses torch.arange per length.
            batch_size (int, optional): Max batch size for same-length proteins. If None, estimate from cuda_memory_GB.
            cuda_memory_GB (int, optional): Used to estimate batch size when batch_size is None.
            stop_grad (bool, optional): If True, disables gradients (faster inference).
            aatypes (List[torch.Tensor], optional): Per-residue amino acid type indices; required if the checkpoint expects aatype input.

        Returns:
            List[dict]: List of dictionaries containing the model outputs in the same order as the input translations.
        """
        # init a progbar that is updated by the batchsize after each batch:
        if self.progress_bar:
            progbar = tqdm(translations, desc='Predicting', unit='proteins')
        else:
            progbar = None
        # the lengths of all input proteins
        lengths = np.array([len(t) for t in translations])

        if res_idx is None:
            res_idx = [torch.arange(len(tensor)) for tensor in translations]
        assert len(res_idx) == len(translations) == len(rotations), f'Input lists must have the same length. Got {len(translations)}, {len(rotations)}, {len(res_idx)}.'
        assert all([len(t) == len(r) == len(idx) for t, r, idx in zip(translations, rotations, res_idx)]), f'All translations, rotations and res_idx must have the same length (the number of residues).'

        self.model.eval().to(self.device)

        model_outputs = [None for _ in range(len(lengths))]

        for length in np.unique(lengths):
            idxs = np.where(lengths == length)[0]
            B = batch_size if batch_size is not None else estimate_max_batchsize(length, memory_GB=cuda_memory_GB)
            # split idxs in parts of approx equal size smaller than B:
            n_splits = int(np.ceil(len(idxs)/B))
            splits = np.array_split(idxs, n_splits)

            for split in splits:
                translations_ = torch.stack([translations[i] for i in split], dim=0)
                rotations_ = torch.stack([rotations[i] for i in split], dim=0)
                res_mask = torch.ones_like(translations_[...,0]).float()
                res_idx_ = torch.stack([res_idx[i] for i in split], dim=0)
                model_input = {'trans_1': translations_.to(self.device), 'rotmats_1': rotations_.to(self.device), 'res_mask': res_mask.to(self.device), 'res_idx': res_idx_.to(self.device)}

                # Add aatype if provided
                if aatypes is not None:
                    aatype_ = torch.stack([aatypes[i] for i in split], dim=0)
                    model_input['aatype'] = aatype_.to(self.device)
                
                if stop_grad:
                    with torch.no_grad():
                        model_output = self.model(model_input)
                else:
                    model_output = self.model(model_input)

                if self.progress_bar:
                    progbar.update(len(split))

                # unbatch:
                for i, idx in enumerate(split):
                    model_outputs[idx] = {k: v[i] for k,v in model_output.items()}

        # assert that all model_outputs are filled
        assert all([m is not None for m in model_outputs]), f'Internal error: not all model_outputs are filled.'

        # now remove trans, rotmats, resmask and the confidence intervals if not requested
        def is_ci(k):
            if not self.confidence_intervals:
                return False
            return 'min_95' in k or 'max_95' in k
        
        for i in range(len(model_outputs)):
            model_outputs[i] = {k: v for k,v in model_outputs[i].items() if k not in ['trans', 'rotmats', 'res_mask'] and not is_ci(k)}

        return model_outputs

    def predict(self, input_path: Union[str, Path], output_folder:str=None, batch_size: int = None, cuda_memory_GB: int = 8, stop_grad: bool = True, path_batchsize: int = 1000, overwrite: bool = False):

        """
        Run inference on a path or folder and write outputs to disk.

        Input paths can be a directory of PDB/CIF files, a single PDB/CIF file,
        or a CSV pointing to processed npz/pkl files. For PDB/CIF input, the
        selected profile is written to the B-factor field.
        Checks whether data is present and only overwrites if `overwrite=True`.
        Args:
            batch_size (int, optional): Maximum batch size. Defaults to None.
            cuda_memory_GB (int, optional): Available GPU memory in GB. Defaults to 8. Reduce if you run out of GPU memory.
            stop_grad (bool, optional): If True, disables gradient computation. Defaults to True.
            path_batchsize (int, optional): Number of paths to process per predict() call. Defaults to 1000. Reduce if you run out of cpu-RAM.
        """

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path {input_path} does not exist.")
        
        input_path = Path(input_path)

        paths, input_ext = parse_input_paths(input_path)
        batched_idxs = list(range(0, len(paths), path_batchsize))

        if overwrite:
            output_folder = input_path if input_path.is_dir() else input_path.parent if output_folder is None else Path(output_folder)
            logging.info(f'Overwrite is set to True. Rewriting input files in {output_folder}.')
        else:
            if output_folder is None:
                output_folder = input_path / 'inference_results' if input_path.is_dir() else input_path.parent / 'inference_results'
                logging.info(f'Overwrite is False. Results will be saved in {output_folder}.')
            else:
                output_folder = Path(output_folder)
                logging.info(f'Output folder is set to {output_folder}. Results will be saved there.')
        
        os.makedirs(output_folder, exist_ok=True)
    
        is_pdb = input_ext == ".pdb"
        is_pkl = input_ext == ".pkl"
        is_npz = input_ext == ".npz"

        logging.info(f"Running inference for {len(paths)} files with extension {input_ext}.")
        for i in batched_idxs:
            batch_paths = paths[i:i + path_batchsize]
            translations, rotations, output_paths, aatypes = [], [], [], []
            
            for path in batch_paths:
                if is_pdb:
                    model_input = frames_from_pdb(pdb_path=path)
                elif is_npz or is_pkl:
                    try:
                        model_input = read_path(path=path)
                    except Exception as e:
                        logging.error(f"Error reading {path}: {e}")
                        continue
                translations.append(model_input['trans_1'])
                rotations.append(model_input['rotmats_1'])
                output_paths.append(path)

                if self.use_aatype:
                    assert "aatype" in model_input, f"'aatype' required by model but missing in {path}"
                    aatypes.append(model_input["aatype"])
                else:
                    aatypes.append(None)
            predictions = self.predict_from_frames(translations=translations,
                                                    rotations=rotations,
                                                    aatypes=aatypes if self.use_aatype else None,                                                    batch_size=batch_size,
                                                    cuda_memory_GB=cuda_memory_GB,
                                                    stop_grad=stop_grad)
            
            if is_pdb:
                logging.info(f"Writing prediction of {self.rmsf_type} as b-factor into {output_folder}...")
            for path, prediction in zip(output_paths, predictions):
                save_prediction(input_path = path,
                                prediction = prediction,
                                output_folder = output_folder,
                                overwrite = overwrite,
                                rmsf_type = self.rmsf_type)

    def predict_from_csv(self, csv_path: Union[str, Path], batch_size: int = None, cuda_memory_GB: int = 8, stop_grad: bool = True, path_batchsize: int = 1000):
        """
        Run inference on a CSV of processed paths and return predictions with ground truth.

        The CSV is expected to contain a processed_path column (via
        `parse_input_paths`). Ground-truth arrays are collected from the
        processed files for keys that overlap with model outputs (typically
        global_rmsf/local_flex).

        Returns:
            predictions (List[dict]): model outputs per entry.
            ground_truth (List[dict]): ground-truth arrays per entry for keys present in predictions.
        """
        total_time = 0
        prediction_time = 0

        start_time = time.time()
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        paths, input_ext = parse_input_paths(csv_path)
        if input_ext not in {".npz"} | {'.pkl', '.pickle', '.pck', '.db', '.pck'}:
            raise ValueError(f"CSV entries must point to .npz or pickle files. Got {input_ext}")

        batched_idxs = list(range(0, len(paths), path_batchsize))
        predictions_all: List[dict] = []
        ground_truth_all: List[dict] = []

        for i in batched_idxs:
            batch_paths = paths[i:i + path_batchsize]
            translations, rotations, aatypes = [], [], []
            gt_batch: List[dict] = []

            for path in tqdm(batch_paths, desc='Loading data', disable=not self.progress_bar):
                model_input, feats = read_path(path=path, return_feats=True)
                translations.append(model_input['trans_1'])
                rotations.append(model_input['rotmats_1'])

                gt = {}
                for key in ['global_rmsf', 'local_flex']:
                    if key in feats:
                        gt[key] = feats[key]
                if gt == {}:
                    warnings.warn(f'Neither global_rmsf nor local_flex found in ground truth.')
                gt_batch.append(gt)

                if self.use_aatype:
                    assert "aatype" in model_input, f"'aatype' required by model but missing in {path}"
                    aatypes.append(model_input["aatype"])
                else:
                    aatypes.append(None)

            start_prediction = time.time()

            predictions = self.predict_from_frames(
                translations=translations,
                rotations=rotations,
                aatypes=aatypes if self.use_aatype else None,
                batch_size=batch_size,
                cuda_memory_GB=cuda_memory_GB,
                stop_grad=stop_grad,
            )

            prediction_time += time.time() - start_prediction

            predictions_all.extend(predictions)
            ground_truth_all.extend(gt_batch)

        total_time = time.time() - start_time
        print(f"Total inference time (including I/O) for {len(paths)} entries: {total_time:.2f} seconds.")
        print(f"Total prediction time: {prediction_time:.2f} seconds.")
        print(f"Average prediction time per entry: {prediction_time / len(paths):.2f} seconds.")

        return predictions_all, ground_truth_all
