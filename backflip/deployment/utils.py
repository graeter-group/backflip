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
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io import load_structure
import warnings
import gzip
import os

from openfold.data.data_transforms import atom37_to_frames
from openfold.utils.rigid_utils import Rigid

from backflip.data.residue_constants import restype_3to1, restype_order_with_x
import biotite.structure.io.pdb as pdb
import numpy as np
from pathlib import Path

from openfold.data import data_transforms
from openfold.utils import rigid_utils

from backflip.data import utils as du
from backflip.data.pdb_dataloader import PICKLE_EXTENSIONS
from typing import Union
from backflip.data.flexibility_utils import batched_rmsf_from_covar


FLEX_FEATS = ['global_rmsf', 'local_flex']
LATEST_TAG = 'backflip-2.1'

CKPT_URLS = {
    'backflip-0.1': 'https://keeper.mpdl.mpg.de/f/e96cda3b3dbd4911af48/?dl=1',
    'backflip-0.2': 'https://keeper.mpdl.mpg.de/f/34c3c08ef8a443bfa5c6/?dl=1',
    'backflip-1.0': 'https://keeper.mpdl.mpg.de/f/d30eb00c17e14f8d86cf/?dl=1', # uses no sequence information
    'backflip-1.0-seq': 'https://keeper.mpdl.mpg.de/f/e5494325fabb47479451/?dl=1', # uses one-hot encoded amino acid type (but no evolutionary information)
    'backflip-2.1': 'https://keeper.mpdl.mpg.de/f/5f914e56becc476eaa16/?dl=1', # ATLAS; outputs per_res_covariance, pairwise_couplings, pairwise_DCCM
    'backflip-2.1-mdcath': 'https://keeper.mpdl.mpg.de/f/d41f98e94ff34cc282ba/?dl=1', # mdCATH (foldseek split)
    'backflip-2.1-joint': 'https://keeper.mpdl.mpg.de/f/bb277fb3190647f0bfb6/?dl=1', # joint ATLAS + mdCATH split
}

CONFIG_URLS = {
    'backflip-0.1': 'https://keeper.mpdl.mpg.de/f/d21a10157fc049928afb/?dl=1',
    'backflip-0.2': 'https://keeper.mpdl.mpg.de/f/c27f1c32892a42c59736/?dl=1',
    'backflip-1.0': 'https://keeper.mpdl.mpg.de/f/1bfa184fbee247c98afa/?dl=1',
    'backflip-1.0-seq': 'https://keeper.mpdl.mpg.de/f/ec0b00204c774c63802a/?dl=1',
    'backflip-2.1': 'https://keeper.mpdl.mpg.de/f/307c0547d4044a8b8d72/?dl=1',
    'backflip-2.1-mdcath': 'https://keeper.mpdl.mpg.de/f/e9311375835b4f919147/?dl=1',
    'backflip-2.1-joint': 'https://keeper.mpdl.mpg.de/f/cf315ff5c58f4b899399/?dl=1',
}

chain_int_mapping = {ch:i for i, ch in enumerate(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))}

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
                       clean_hetero: bool = False) -> tuple:
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
    noncanonicals = {"MSE": None,"TPO": None,"MLY": None,"CME": None,"PTR": None,
                     "SEP": None,"SAH": None,"CSO": None,"PCA": None,"KCX": None,
                     "CAS": None,"CSD": None,"MLZ": None,"OCS": None,"ALY": None,
                     "CSS": None,"CSX": None,"HIC": None,"HYP": None,"YCM": None,
                     "YOF": None,"M3L": None,"PFF": None,"CGU": None,"FTR": None,
                     "LLP": None,"CAF": None,"CMH": None,"MHO": None,"MHS": None}
    if compressed:
        with gzip.open(pdb_loc, "rt") as file_handle:
            pdb_file = pdb_io.PDBFile.read(file_handle)
            protein = pdb_io.get_structure(pdb_file, extra_fields=['b_factor'])
    else:
        protein = load_structure(pdb_loc, extra_fields=["b_factor"])

    if clean_hetero:
        is_stack = isinstance(protein, bt.structure.AtomArrayStack)
        hetero_mask = protein.hetero.copy()
        hetero_protein = protein[:, hetero_mask] if is_stack else protein[hetero_mask]  # Get hetero atoms
        # check if any hetero atoms are non-canonical residues
        noncanonical_hetero_mask = np.isin(hetero_protein.res_name, list(noncanonicals.keys()))
        if np.any(noncanonical_hetero_mask):
            print(f"Warning: Found non-canonical residues in heteroatoms of {pdb_loc}: {np.unique(hetero_protein.res_name[noncanonical_hetero_mask])}. Keeping them in the structure.")
            hetero_mask[hetero_mask] = ~noncanonical_hetero_mask  # Keep non-canonical hetero atoms
        protein = protein[:, ~hetero_mask] if is_stack else protein[~hetero_mask]

    if isinstance(protein, bt.structure.AtomArrayStack):
        if not ensemble:
            print(f"Multiple models found in {pdb_loc}. Taking first model.")
            protein = protein[0]
    elif isinstance(protein, bt.structure.AtomArray):
        if ensemble:
            print(f"Only a single model found in {pdb_loc}. Returning it as-is.")
    elif ensemble:
        print(f"Only a single model found in {pdb_loc}. Returning it as-is.")

    mask = (~protein.hetero) & (protein.ins_code == '')
    if backbone:
        mask &= np.isin(protein.atom_name, ['N', 'CA', 'C', 'O'])
    if isinstance(protein, bt.structure.AtomArrayStack):
        protein = protein[:, mask]
    else:
        protein = protein[mask]

    chain_ids = np.unique(protein.chain_id)
    n_atoms = protein.array_length() if isinstance(protein, bt.structure.AtomArrayStack) else len(protein)
    if n_atoms == 0:
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

def save_tite_as_cif(protein_array, pdb_name:str, out_folder:str)->None:
    '''
    Saves the selected chain as a clean .cif file with renumbered residues
    '''
    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, protein_array, data_block=pdb_name)
    loc_cif_save = os.path.join(out_folder, f'{pdb_name}.cif')
    cif_file.write(loc_cif_save)
    return None

def backbone_to_frames(N_atoms, CA_atoms, C_atoms, resnames, chain_ids=None):
    seq_numerical = torch.tensor([restype_order_with_x.get(aa, restype_order_with_x['X']) for aa in resnames])
    seq_onehot = torch.nn.functional.one_hot(seq_numerical, 21).float()

    N = len(N_atoms)
    X = torch.zeros(N, 37, 3)
    X[:, 0] = torch.tensor(N_atoms)
    X[:, 1] = torch.tensor(CA_atoms)
    X[:, 2] = torch.tensor(C_atoms)
    X -= torch.mean(torch.tensor(CA_atoms), dim=0)
    atom_mask = torch.zeros((N, 37)).double()
    atom_mask[:, :3] = 1
    
    protein = {
        "aatype": seq_numerical.long(),
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
        "aatype": seq_numerical,
        "chain_ids": chain_ids if chain_ids is not None else torch.zeros(N, dtype=torch.int64)
    }

def _res_keys(atoms):
    return list(zip(atoms.chain_id.tolist(), atoms.res_id.tolist(), atoms.ins_code.tolist()))

def drop_incomplete_backbone_residues(structure, source_desc: str = ""):
    """
    Drops residues with partially resolved backbone density (missing N, CA and/or C),
    matched per-residue by (chain_id, res_id, ins_code).
    """

    n_keys = set(_res_keys(structure[structure.atom_name == 'N']))
    ca_keys = set(_res_keys(structure[structure.atom_name == 'CA']))
    c_keys = set(_res_keys(structure[structure.atom_name == 'C']))
    complete_keys = n_keys & ca_keys & c_keys
    incomplete_keys = (n_keys | ca_keys | c_keys) - complete_keys

    if incomplete_keys:
        suffix = f" in {source_desc}" if source_desc else ""
        print(f"Warning: dropping {len(incomplete_keys)} residue(s) with incomplete backbone (missing N, CA and/or C){suffix}: {sorted(incomplete_keys)}")

    keep_mask = np.array([k in complete_keys for k in _res_keys(structure)])
    return structure[keep_mask]

def frames_from_pdb_tite(pdb_path:Path):
    # NOTE: this might likely cause issues if there are HETATM that still have bb atoms. We assume that the PDB files used for inference are clean and only contain ATOM records for the backbone atoms.
    structure, chain_ids = get_structure_tite(pdb_path, backbone=False, ensemble=False, clean_hetero=True)
    # NOTE: important with some pdbs that don't have all backbone atoms resolved
    structure = drop_incomplete_backbone_residues(structure, source_desc=str(pdb_path))

    N_atoms = structure[structure.atom_name == 'N'].coord
    CA_atoms = structure[structure.atom_name == 'CA'].coord
    C_atoms = structure[structure.atom_name == 'C'].coord
    ca_atoms_all = structure[structure.atom_name == 'CA']
    res_names = ca_atoms_all.res_name
    seq = np.array([restype_3to1.get(r, 'X') for r in res_names])
    if np.any(seq == 'X'):
        unknown = np.unique(res_names[seq == 'X']).tolist()
        # TODO: the aatype embedding (node_embedder.py) hardcodes num_classes=20 (no 'X'/unknown
        # class), so it can't accept token 20. Until that's fixed to support an explicit unknown
        # class, fall back to treating non-canonical residues as ALA so inference doesn't crash.
        print(f"Warning: non-canonical residue(s) {unknown} in {pdb_path} have no canonical 1-letter mapping; treating them as ALA ('A') for the aatype embedding.")
        seq[seq == 'X'] = 'A'
    res_idx = ca_atoms_all.res_id
    chain_ids = torch.tensor(np.array([chain_int_mapping[ch] for ch in ca_atoms_all.chain_id]))
    data = backbone_to_frames(N_atoms, CA_atoms, C_atoms, seq, chain_ids=chain_ids)
    return data, seq

def frames_from_pdb(pdb_path:Path)->dict[str, torch.Tensor]:
    """
    Extracts frames from a PDB file. If the PDB file contains multiple states, it will only use the first state.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        Tuple of trans, rotmats, seq_onehot.
    """
    pdb_path = str(pdb_path)
    eq = mdtraj.load(pdb_path)
    N_atoms = eq.xyz[0, eq.top.select('name N'), :] * 10
    CA_atoms = eq.xyz[0, eq.top.select('name CA'), :] * 10
    C_atoms = eq.xyz[0, eq.top.select('name C'), :] * 10
    seq = np.array(list("".join(eq.top.to_fasta())))
    data = backbone_to_frames(N_atoms, CA_atoms, C_atoms, seq)
    return data, seq

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

def save_prediction(input_path, prediction, output_folder, overwrite=False, rmsf_as_bfactor=False):

    """
    Saves all predicted edge features into a separate .npz file in the output folder.
    If the input file is a .npz or pickle file, it will save the predictions into the same file (overwriting existing keys if overwrite=True) and save it to the output folder.
    If the input file is a PDB/CIF and rmsf_as_bfactor=True, also writes a .cif file with
    the isotropic RMSF (derived from the predicted per-residue covariance) in the B-factor column.

    Args:
        input_path: Path to the input file (PDB/CIF or .npz/.pkl).
        prediction: Dictionary of predicted features.
        output_folder: Folder to save the output files.
        overwrite: Whether to overwrite existing keys in .npz/.pkl files. Default is False.
        rmsf_as_bfactor: If True and input_path is a PDB/CIF file, additionally writes a .cif
            file with the isotropic RMSF (10 * sqrt(trace(per_res_covariance))) in the B-factor
            column. Default is False.
    """
    
    PICKLE_EXTENSIONS = {'.pkl', '.pickle', '.pck', '.db', '.pck'}
    PRED_FEATS = {'per_res_covariance', 'pairwise_couplings', 'pairwise_DCCM'}
    file_extension = Path(input_path).suffix
    if file_extension not in {'.npz', '.pdb', '.cif'} | PICKLE_EXTENSIONS:
        raise ValueError(f'Unknown file extension {file_extension}')

    if file_extension in {'.pdb', '.cif'}:
        # TODO: this will raise issues if there are HETATM that still have bb atoms.
        loaded_data, _ = get_structure_tite(input_path, ensemble=False, backbone=True, clean_hetero=True)
        # Match frames_from_pdb_tite: drop residues with partially resolved backbone density
        # (missing N, CA and/or C) so the N/CA/C/O counts below stay consistent per residue.
        loaded_data = drop_incomplete_backbone_residues(loaded_data, source_desc=str(input_path))
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
        # saving edge predictions as an npz file
        npz_folder = os.path.join(output_folder, 'npz')
        os.makedirs(npz_folder, exist_ok=True)
        npz_out = os.path.join(npz_folder, f'{Path(input_path).stem}_pred.npz')
        preds_all = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in prediction.items() if k in PRED_FEATS}
        np.savez(npz_out, **preds_all)

        if rmsf_as_bfactor:
            per_res_cov = prediction['per_res_covariance']
            # batched_rmsf_from_covar returns (B, N)
            rmsf_pred = np.asarray(batched_rmsf_from_covar(per_res_cov)).reshape(-1)

            if rmsf_pred.shape[0] != n_residues:
                raise ValueError(
                    f"Predicted per-residue RMSF length ({rmsf_pred.shape[0]}) does not match "
                    f"the number of residues ({n_residues}) for {input_path}."
                )

            # broadcast per-residue RMSF to backbone atoms (N, CA, C, [O optional on last residue])
            idx = np.arange(n_residues).repeat(4)
            if O == n_residues - 1:
                # exactly one missing O, assumed to be the last atom in the file
                idx = idx[:-1]
            
            b_factors = np.asarray(rmsf_pred[idx], dtype=np.float32)
            if b_factors.shape[0] != loaded_data.array_length():
                raise ValueError(
                    f"Broadcast b-factor length ({b_factors.shape[0]}) does not match the number "
                    f"of backbone atoms ({loaded_data.array_length()}) for {input_path}."
                )
            
            out_folder_cifs = os.path.join(output_folder, 'cifs')
            os.makedirs(out_folder_cifs, exist_ok=True)
            loaded_data.set_annotation('b_factor', b_factors)
            save_tite_as_cif(loaded_data, pdb_name=Path(input_path).stem, out_folder=out_folder_cifs)

def profile_from_bfac(path:str):
    struct = load_structure(path, extra_fields=["b_factor"])
    bfacs_CA = struct[struct.atom_name == 'CA'].b_factor
    return bfacs_CA
