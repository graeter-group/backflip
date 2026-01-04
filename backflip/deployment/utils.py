# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import numpy as np
from pathlib import Path
import requests
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any
import torch
import mdtraj
import biotite as bt
import biotite.structure.io.pdb as pdb_io
from biotite.structure.io import load_structure
import warnings
import gzip
import os

from openfold.data.data_transforms import atom37_to_frames
from openfold.utils.rigid_utils import Rigid

from gafl.data.residue_constants import restype_order, restype_3to1, restype_order_with_x

import biotite.structure.io.pdb as pdb
import numpy as np
from pathlib import Path

from openfold.data import data_transforms
from openfold.utils import rigid_utils

from backflip.data import utils as du
from backflip.data.pdb_dataloader import PICKLE_EXTENSIONS
from typing import Union

FLEX_FEATS = ['global_rmsf', 'local_flex']

LATEST_TAG = 'backflip-1.0'
CKPT_URLS = {
    'backflip-0.1': 'https://keeper.mpdl.mpg.de/f/e96cda3b3dbd4911af48/?dl=1',
    'backflip-0.2': 'https://keeper.mpdl.mpg.de/f/34c3c08ef8a443bfa5c6/?dl=1',
    'backflip-1.0': 'https://keeper.mpdl.mpg.de/f/d30eb00c17e14f8d86cf/?dl=1', # uses no sequence information
    'backflip-1.0-seq': 'https://keeper.mpdl.mpg.de/f/e5494325fabb47479451/?dl=1', # uses one-hot encoded amino acid type (but no evolutionary information)
}
CONFIG_URLS = {
    'backflip-0.1': 'https://keeper.mpdl.mpg.de/f/d21a10157fc049928afb/?dl=1',
    'backflip-0.2': 'https://keeper.mpdl.mpg.de/f/c27f1c32892a42c59736/?dl=1',
    'backflip-1.0': 'https://keeper.mpdl.mpg.de/f/1bfa184fbee247c98afa/?dl=1',
    'backflip-1.0-seq': 'https://keeper.mpdl.mpg.de/f/ec0b00204c774c63802a/?dl=1',
}


def estimate_max_batchsize(n_res, memory_GB=8):
    """
    Estimate the maximum batch size that can be used for sampling. Hard-coded from empiric experiments. We found a dependency that is inversely proportional to the number of residues in the protein.
    """
    if not isinstance(n_res, np.ndarray):
        n_res = np.array([n_res])
    A = 1e6
    B = 40
    batchsize = A/(n_res+B)**2 * memory_GB
    batchsize = np.floor(batchsize)
    ones = np.ones_like(batchsize)
    out = np.max(np.stack([batchsize, ones], axis=0), axis=0)
    if out.shape == (1,):
        return int(out[0])
    return out.astype(int)

# overwrite args that are specified in cfg:
def recursive_update(cfg:dict, cfg_:dict):
    for key in cfg.keys():
        if key in cfg_.keys():
            if isinstance(cfg[key], dict):
                recursive_update(cfg[key], cfg_[key])
            else:
                cfg_[key] = cfg[key]

def get_root_dir()->Path:
    """
    Get the root directory of the package.
    """
    return Path(__file__).parent.parent.parent


def ckpt_path_from_tag(tag:str='latest'):
    """
    Get the path to the checkpoint assumed to be located at root_dir/models/tag/*.ckpt. Checks existence and uniqueness of the checkpoint file and existence of the config file.
    """

    if tag == 'latest':
        tag = LATEST_TAG

    root_dir = get_root_dir()
    ckpt_dir = root_dir / 'models' / tag

    if not ckpt_dir.exists():
        if not tag in CKPT_URLS.keys():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found and {tag} not found in the hard-coded URLs for downloading.")
        else:
            ckpt_path = download_model(tag)
            return ckpt_path
    
    ckpt_files = list(ckpt_dir.glob('*.ckpt'))
    if len(ckpt_files) == 0:
        if not tag in CKPT_URLS.keys():
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir} and {tag} not found in the hard-coded URLs for downloading.")
        else:
            ckpt_path = download_model(tag)
            return ckpt_path
    elif len(ckpt_files) > 1:
        raise FileNotFoundError(f"Multiple checkpoint files found in {ckpt_dir}.")
    if not (ckpt_dir/'config.yaml').exists():
        raise FileNotFoundError(f"No config file found in {ckpt_dir}.")

    return ckpt_files[0]

def _download_file(url:str, target_path:Path, progress_bar:bool=True):
    # Start the download
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful

    if progress_bar:
        # Get the total file size from headers
        total_size = int(response.headers.get('content-length', 0))

        # Initialize the progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True) as t:
            with open(target_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024*2**2):
                    file.write(chunk)
                    t.update(len(chunk))
    else:
        with open(target_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024*2**2):
                file.write(chunk)

    assert target_path.exists(), f"Download failed, file not found at {target_path}"


def download_model(tag:str='latest'):
    """
    Download the model checkpoint and config file from the hard-coded URLS.
    """
    if tag == 'latest':
        tag = LATEST_TAG

    root_dir = get_root_dir()
    ckpt_dir = root_dir / 'models' / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_url = CKPT_URLS[tag]
    config_url = CONFIG_URLS[tag]

    ckpt_path = ckpt_dir / f'{tag}.ckpt'
    config_path = ckpt_dir / 'config.yaml'

    _download_file(config_url, config_path, progress_bar=False)
    print(f"Downloading model weights for tag '{tag}' from {ckpt_url}")
    _download_file(ckpt_url, ckpt_path, progress_bar=True)

    return ckpt_path

def get_structure_tite(pdb_loc: str,
                       compressed=False,
                       backbone: bool = True,
                       ensemble: bool = False,
                       clean_hetero: bool = True) -> tuple:
    '''
    Load a protein structure from a PDB or CIF file using Biotite.

    Args:
        pdb_loc (str): Path to the structure file.
        compressed (bool): Whether the file is gzipped.
        backbone (bool): Return only backbone atoms if True.
        ensemble (bool): Load full ensemble if True.
        clean_hetero (bool): Remove heteroatoms if True.

    Returns:
        protein (AtomArray): Filtered protein structure.
        chain_id (str | list): Chain ID(s).
    '''
    EXCLUDE_HETERO = {
        "HOH", "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "ZN", "NO3", "ACT", "DTT", "TRS", "BME",
        "GOL", "EDO", "MN", "CO", "CU", "FE", "NI", "CO3", "FES", "FAD", "FMN", "PCA", "PLM",
        "PQQ", "PQQH2", "PDS", "PPT", "PQQH", "FADH2", "FADH"
    }

    if compressed:
        with gzip.open(pdb_loc, "rt") as file_handle:
            pdb_file = pdb_io.PDBFile.read(file_handle)
            protein = pdb_io.get_structure(pdb_file, extra_fields=['b_factor'])
    else:
        protein = load_structure(pdb_loc, extra_fields=["b_factor"])

    if isinstance(protein, bt.structure.AtomArrayStack):
        print(f"Multiple models found in {pdb_loc}. Taking first model.")
        protein = protein[0]

    if ensemble:
        mask = (~protein.hetero) & (protein.ins_code == '')
        if backbone:
            mask &= np.isin(protein.atom_name, ['N', 'CA', 'C', 'O'])
        protein = protein[:, mask]
        return protein, list(np.unique(protein.chain_id))

    # Single-structure mode
    chain_ids = np.unique(protein.chain_id)
    if clean_hetero:
        protein = protein[
            (~protein.hetero) &
            (protein.ins_code == '')
        ]
    else:
        protein = protein[
            (protein.ins_code == '') &
            (~np.isin(protein.res_name, list(EXCLUDE_HETERO)))
        ]

    if backbone:
        protein = protein[np.isin(protein.atom_name, ['N', 'CA', 'C', 'O'])]

    if len(protein) == 0:
        print(f"No atoms left after filtering in {pdb_loc}")
        return None, None

    return protein, chain_ids.tolist() if len(chain_ids) > 1 else chain_ids[0]

def save_tite_as_pdb(protein_array, pdb_name:str, out_folder:str)->None:
    '''
    Saves the selected chain as a clean .pdb file with renumbered residues
    '''
    file = pdb_io.PDBFile()
    file.set_structure(protein_array)
    loc_pdb_save = os.path.join(out_folder, f'{pdb_name}.pdb')
    file.write(loc_pdb_save)
    return None

def backbone_to_frames(N_atoms, CA_atoms, C_atoms, resnames):
    seq_numerical = torch.tensor([restype_order_with_x[aa] for aa in resnames])
    seq_onehot = torch.nn.functional.one_hot(seq_numerical, 21).float()

    N = len(N_atoms)
    X = torch.zeros(N, 37, 3)
    X[:, 0] = torch.tensor(N_atoms)
    X[:, 1] = torch.tensor(CA_atoms)
    X[:, 2] = torch.tensor(C_atoms)
    X -= torch.mean(torch.tensor(CA_atoms), dim=0)
    aatypes = torch.tensor([restype_order[aa] for aa in resnames]).long()
    atom_mask = torch.zeros((N, 37)).double()
    atom_mask[:, :3] = 1
    protein = {
        "aatype": aatypes,
        "all_atom_positions": X,
        "all_atom_mask": atom_mask,
    }
    frames = atom37_to_frames(protein)
    rigids_0 = Rigid.from_tensor_4x4(frames['rigidgroups_gt_frames'])[:,0]
    trans = rigids_0.get_trans()
    rotmats = rigids_0.get_rots().get_rot_mats()

    return {
        "trans_1": trans,
        "rotmats_1": rotmats,
        "seq_onehot": seq_onehot,
        "aatype": aatypes,
    }


def frames_from_pdb(pdb_path:Path)->dict[str, torch.Tensor]:
    """
    Extracts frames from a PDB file. If the PDB file contains multiple states, it will only use the first state.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        Tuple of trans, rotmats, seq_onehot.
    """

    eq = mdtraj.load(pdb_path)
    N_atoms = eq.xyz[0, eq.top.select('name N'), :] * 10
    CA_atoms = eq.xyz[0, eq.top.select('name CA'), :] * 10
    C_atoms = eq.xyz[0, eq.top.select('name C'), :] * 10
    seq = np.array(list("".join(eq.top.to_fasta())))
    data = backbone_to_frames(N_atoms, CA_atoms, C_atoms, seq)
    return data


def chain_feats_from_pdb(pdb_path: Path) -> dict:
    # Load the PDB file
    pdb_file = pdb.PDBFile.read(pdb_path)
    atom_array = pdb.get_structure(pdb_file)[0]  # Assuming we're only interested in the first model

    # Filter for backbone atoms (N, CA, C)
    mask = np.isin(atom_array.atom_name, ['N', 'CA', 'C'])
    filtered_atoms = atom_array[mask]

    # Organize the data
    residues = filtered_atoms.res_name
    res_ids = filtered_atoms.res_id
    atoms = filtered_atoms.atom_name
    coords = filtered_atoms.coord
    chain_id = np.array(['A'] * len(atoms)).astype(str)
    elements = filtered_atoms.element

    # Create dictionaries for each residue
    chain_features = {
        "res_name": residues,
        "res_id": res_ids,
        "atom_name": atoms,
        "coords": coords,
        "chain_id": chain_id,
        "element": elements,
    }

    return chain_features


def read_path(path: str, seed:int=123, ensemble:bool=False, return_feats:bool=False):
    '''
    Reduced version of the process_csv_row function in the PDBDataLoader class for inference.
    Returns:
    {'rotmats_1': torch.Tensor, 'trans_1': torch.Tensor}
    '''
    path_extension = Path(path).suffix
    if path_extension in PICKLE_EXTENSIONS:
        processed_feats = du.read_pkl(path)
        processed_feats = du.parse_chain_feats(processed_feats)
    elif path_extension == '.npz':
        if ensemble:
            processed_feats, feat_dict = du.read_npz(path, seed=seed, conf_idx=0)
        else:
            processed_feats, feat_dict = du.read_npz(path)
        processed_feats = du.parse_npz_feats(npz_feats=processed_feats)
        # here the actual residue indices which are modeled are stored in the residue_index field
    else:
        raise ValueError(f'Unknown file extension {path_extension}')

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

    if 'bb_mask' in processed_feats and not np.all(processed_feats['bb_mask']==1):
        warnings.warn(f'bb_mask is not all 1s for {path}. This might lead to unexpected results.')

    d = {
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        }
    if 'aatype' in processed_feats:
        d['aatype'] = torch.tensor(processed_feats['aatype']).long()

    for flex_feat in FLEX_FEATS:
        if flex_feat in feat_dict:
            assert feat_dict[flex_feat].shape[0] == d['trans_1'].shape[0], \
                f"Flexibility feature {flex_feat} length {feat_dict[flex_feat].shape[0]} does not match number of residues {d['trans_1'].shape[0]} in {path}."
            processed_feats[flex_feat] = torch.tensor(feat_dict[flex_feat]).float()

    if return_feats:
        return d, processed_feats
    else:
        return d

def parse_input_paths(input_path: Union[str, Path]) -> Tuple[List[Path], str]:
    input_path = Path(input_path)
    PICKLE_EXTENSIONS = {'.pkl', '.pickle', '.pck', '.db', '.pck'}

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Case 1: CSV file
    if input_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(input_path)

        if 'processed_path' not in df.columns:
            raise ValueError("'processed_path' column not found in CSV")

        paths = [Path(p) for p in df['processed_path']]
        exts = set(p.suffix for p in paths)

        if len(exts) > 1:
            raise ValueError("All processed_path entries must have the same file extension.")

        ext = exts.pop()
        if ext not in {'.npz', '.pdb', '.cif'} | PICKLE_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        return paths, ext

    # Case 2: directory
    elif input_path.is_dir():
        paths = list(input_path.glob("*.pdb")) + list(input_path.glob("*.cif"))
        if not paths:
            raise ValueError("No .pdb or .cif files found in directory.")
        return paths, '.pdb'  # could be '.cif' too, doesn't matter for mixed .pdb/.cif use

    # Case 3: single file
    elif input_path.is_file():
        if input_path.suffix not in {'.pdb', '.cif'}:
            raise ValueError("Only .pdb or .cif files are supported as single file input.")
        return [input_path], input_path.suffix

    raise ValueError("Invalid input path")

def save_prediction(input_path, prediction, output_folder, overwrite=False, rmsf_type:str='global_rmsf'):
    
    PICKLE_EXTENSIONS = {'.pkl', '.pickle', '.pck', '.db', '.pck'}
    file_extension = Path(input_path).suffix
    if file_extension not in {'.npz', '.pdb', '.cif'} | PICKLE_EXTENSIONS:
        raise ValueError(f'Unknown file extension {file_extension}')

    if file_extension in {'.pdb', '.cif'}:
        # TODO: this will raise issues if there are HETATM that still have bb atoms.
        loaded_data, _ = get_structure_tite(input_path, ensemble=False, backbone=True, clean_hetero=False)
    elif file_extension in {'.npz'} | PICKLE_EXTENSIONS:
        if file_extension == '.npz':
            loaded_data = dict(np.load(input_path))
        elif file_extension in PICKLE_EXTENSIONS:
            loaded_data = du.read_pkl(input_path)
        else:
            raise ValueError(f'Unsupported file extension {file_extension} for input path {input_path}')
        # simpler to just write into the output folder
        for k, v in prediction.items():
            if not overwrite and k in loaded_data:
                raise ValueError(f'Key {k} already present in {input_path}. Set overwrite=True to overwrite existing keys.')
            loaded_data[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
    else:
        raise ValueError(f'Unsupported file extension {file_extension} for input path {input_path}')
    
    if file_extension in {'.npz'}:
        np.savez(Path(output_folder) / Path(input_path).name, **loaded_data)
    elif file_extension in PICKLE_EXTENSIONS:
        du.write_pkl(Path(output_folder) / Path(input_path).name, loaded_data)
    else:
        # Counts only; assume backbone order per residue (N, CA, C, [O optional on last residue])
        N  = int(np.sum(loaded_data.atom_name == 'N'))
        CA = int(np.sum(loaded_data.atom_name == 'CA'))
        C  = int(np.sum(loaded_data.atom_name == 'C'))
        O  = int(np.sum(loaded_data.atom_name == 'O'))

        # Require N, CA, C present for every residue; allow O to be missing only for the last residue; this happens sometimes
        if not (N == CA == C and O in (N, N - 1)):
            raise ValueError(
                f"Unexpected backbone counts for {input_path}: N={N}, CA={CA}, C={C}, O={O}"
            )

        n_residues = N  # residues inferred from N/CA/C
		
        pred = prediction[rmsf_type]
        pred_np = pred.detach().cpu().numpy().astype(np.float32) if isinstance(pred, torch.Tensor) \
                else np.asarray(pred, dtype=np.float32)

        length = pred_np.shape[0]
        if length not in (n_residues, n_residues + 1):
            raise ValueError(
                f"Per-residue prediction length ({length}) not in {{n_residues, n_residues+1}} "
                f"(n_residues={n_residues}) for {input_path}."
            )
        if length == n_residues + 1:
            pred_np = pred_np[:n_residues]

        idx = np.arange(n_residues).repeat(4)
        if O == n_residues - 1:
            # exactly one missing O, assumed to be the last atom in the file
            idx = idx[:-1]

        b_factors = pred_np[idx]
        b_factors = np.asarray(b_factors, dtype=np.float32).reshape(-1)

        if b_factors.ndim != 1:
            raise ValueError(f"b_factors not 1-D after conversion: shape={b_factors.shape}")
        
        loaded_data.set_annotation('b_factor', b_factors)
        save_tite_as_pdb(loaded_data, pdb_name=Path(input_path).stem, out_folder=output_folder)

def profile_from_bfac(path:str):
    struct = load_structure(path, extra_fields=["b_factor"])
    bfacs_CA = struct[struct.atom_name == 'CA'].b_factor
    return bfacs_CA
