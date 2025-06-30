# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import mdtraj as md
import numpy as np
from pathlib import Path
from typing import List, Union
import biotite.structure.io as strucio
from biotite.structure.io.xtc import XTCFile
from biotite.structure.superimpose import superimpose, superimpose_apply
from biotite.structure import AtomArrayStack
from tqdm.auto import tqdm
from pathlib import Path
import mdtraj as md

def compute_rmsf(tite, window_size:int=13, n_ref:int=10, n_draw:int=100, np_seed:int=42):
    """
    Compute RMSF of a trajectory using local structural alignment.
    Arguments:
    ----------
    tite : biotite.structure.AtomArrayStack
        The trajectory to compute the local RMSF for.
    window_size : int
    The size of the sliding window for local alignment expressed (this is a number of residues). If 0 or not finite, global alignment is performed.
    n_ref : int
    The number of reference conformations to use for the alignment.
    n_draw : int
    Number of conformations to randomly draw to align with each of the n_ref reference conformations.
    np_seed : int
    The seed for the random number generator.
        """
    alignment = get_alignment(confs=tite,
                         global_alignment=not np.isfinite(window_size) or window_size <= 0, 
                         window_size=window_size, 
                         n_ref=n_ref, 
                         n_draw=n_draw, 
                         np_seed=np_seed)
    rmsfs = calc_rmsfs(pos=alignment[0], ref=alignment[1], convolution_window_size=1)
    return rmsfs

def load_trajectory(pdb_path:Union[str,Path], xtc_path:Union[List,Union[str,Path]]=None):
    """
    Load a trajectory and its reference structure from PDB and XTC files, then align all frames to the first frame.
    
    Parameters:
    pdb_path (str): Path to the PDB file used as the reference structure.
    xtc_paths (list): List of paths to XTC files containing the trajectory data.
    
    Returns:
    md.Trajectory: Aligned full trajectory.
    """
    if xtc_path is not None:
        # Load the full trajectory from the XTC file
        if not isinstance(xtc_path, list):
            full_traj = md.load(xtc_path, top=pdb_path)
        else:
            # concat all trajs:
            full_traj = md.load(xtc_path[0], top=pdb_path)
            for xtc in xtc_path[1:]:
                full_traj = full_traj.join(md.load(xtc, top=pdb_path))
    else:
        full_traj = md.load(pdb_path)
    
    return full_traj


def load_ca_trajectory(pdb_path:Union[str,Path], xtc_path:Union[List,Union[str,Path]]=None):
    """
    Load a trajectory and its reference structure from PDB and XTC files, then align all frames to the first frame.
    
    Parameters:
    pdb_path (str): Path to the PDB file used as the reference structure.
    xtc_paths (list): List of paths to XTC files containing the trajectory data.
    
    Returns:
    md.Trajectory: Aligned trajectory containing only C-alpha atoms.
    """
    full_traj = load_trajectory(pdb_path, xtc_path)
    
    # Select only C-alpha atoms
    ca_indices = full_traj.topology.select('name CA')
    ca_traj = full_traj.atom_slice(ca_indices)
    return ca_traj


def backbone_coords_from_traj(traj:md.Trajectory):
    """
    Extract the backbone coordinates from a trajectory, compatible with deployment.utils.chain_feats_from_pdb. Unit: angstroem.
    
    Parameters:
    traj (md.Trajectory): Trajectory object containing the C-alpha atoms.
    
    Returns:
    np.array: Backbone coordinates of the trajectory of shape (n_frames, n_residues*3, 3).
    """
    # Select only backbone atoms (N, CA, C)
    backbone_indices = traj.topology.select('name N or name CA or name C')

    # Slice the trajectory to include only backbone atoms
    backbone_traj = traj.atom_slice(backbone_indices)

    # Extract coordinates
    backbone_coords = backbone_traj.xyz  # This has shape (n_frames, n_selected_atoms, 3)
    # convert from nm to angstroem:
    backbone_coords = backbone_coords*10

    return backbone_coords

def biotite_from_xtc(pdbfile, xtcfile):
    """
    Load a trajectory from a PDB file and an XTC file using biotite.
    """
    # Load the initial structure from the PDB file
    atom_array = strucio.load_structure(str(pdbfile))
    # Open and read the trajectory file
    xtc_file = XTCFile.read(str(xtcfile))
    # Extract coordinate shape
    atom_array_stack = xtc_file.get_structure(template=atom_array)
    return atom_array_stack


def biotite_from_pdb(pdbfile):
    """
    Load a trajectory from a PDB file using biotite.
    """
    atom_array = strucio.load_structure(str(pdbfile))
    return atom_array


def get_alignment(confs:AtomArrayStack,
                  global_alignment: bool = False,
                  window_size: int = 13,
                  n_ref: int = 10,
                  n_draw: int = 10,
                  np_seed=123,
                  store_full_backbone: bool = False):
    """
    Align the positions of alpha carbons (CA atoms) in a trajectory to a set of reference conformations.

    Parameters:
    - confs: AtomArrayStack containing the trajectory conformations.
    - global_alignment: If True, align the whole protein instead of a sliding window.
    - window_size: Size of the sliding window for local alignment. If 0 or not finite, global alignment is performed.
    - n_ref: Number of reference conformations to select for alignment.
    - n_draw: Number of conformations to randomly draw to align with each of the n_ref reference conformations.
    - np_seed: Seed for random number generation to ensure reproducibility.
    - store_full_backbone: If True, store the full backbone atoms (N, CA, C, O) instead of just CA.
    
    Returns:
    - aligned_positions: np.ndarray(float) of shape (n_ref, N_residues, n_draw, 3)
    - ref_positions: np.ndarray(float) of shape (n_ref, N_residues, 3)
    """
    
    np.random.seed(np_seed)
    # Assumes that in default mode local alignment is performed with a window size of 13, as reported in paper.
    calc_global_rmsf = False
    if not global_alignment and window_size <= 0:
        raise ValueError("If 'global_alignment' is False, 'window_size' must be a positive integer.")
    if global_alignment:
        window_size = 0
        calc_global_rmsf = True

    if not isinstance(confs, AtomArrayStack):
        raise TypeError(f"The 'confs' parameter must be an AtomArrayStack, but is {type(confs)}")

    ca_mask = confs[0].atom_name == "CA"
    num_res = int(np.sum(ca_mask))
    allowed_atoms = ["N", "CA", "C", "O"]

    print(f"Number of residues: {num_res}")

    num_frames = len(confs)
    print(f"For each reference, randomly sampling frames: {n_draw} of {num_frames}")

    total_calls = n_ref if calc_global_rmsf else num_res * n_ref
    if calc_global_rmsf:
        progress_bar = tqdm(total=total_calls, desc=f"Globally-aligning frames")
    else:
        progress_bar = tqdm(total=total_calls, desc=f"Locally-aligning frames")

    ref_conf_idxs = np.random.choice(len(confs), n_ref, replace=False)
    ref_confs = [confs[i] for i in ref_conf_idxs]

    aligned_per_ref = []
    refs_per_ref = []

    for ref_idx, ref_conf in enumerate(ref_confs):
        ref_CAs = ref_conf[ref_conf.atom_name == "CA"]
        assert len(ref_CAs) == num_res
        aligned_positions = []
        ref_positions = []

        ref_conf_idx = ref_conf_idxs[ref_idx]
        sample_idxs = np.random.choice(np.delete(np.arange(len(confs)), ref_conf_idx), n_draw, replace=False)
        sampled_confs = [confs[j] for j in sample_idxs]

        for res_idx in range(num_res):
            progress_bar.update(1)
            sample_positions = []

            if not calc_global_rmsf:
                start_index = max(0, res_idx - window_size // 2)
                end_index = min(num_res, res_idx + window_size // 2 + 1)
                res_idx_in_window = res_idx - start_index

                window_ref = ref_CAs[start_index:end_index]

                if not store_full_backbone:
                    ref_positions.append(window_ref[res_idx_in_window].coord)
                else:
                    ref_bb = ref_conf[ref_conf.atom_name.isin(allowed_atoms)]
                    ref_res_bb = ref_bb[ref_bb.res_id == res_idx + 1]
                    ref_positions.append(ref_res_bb.coord)
            else:
                start_index = 0
                end_index = num_res
                window_ref = ref_CAs
                ref_positions = ref_CAs.coord if not store_full_backbone else ref_conf[ref_conf.atom_name.isin(allowed_atoms)].coord

            global_aligned = []

            for conformation in sampled_confs:
                conf_CAs = conformation[conformation.atom_name == "CA"]
                window_conf = conf_CAs[start_index:end_index]

                fitted, transform = superimpose(
                    window_ref, window_conf, atom_mask=(window_ref.atom_name == "CA"))

                if store_full_backbone:
                    conf_bb = conformation[conformation.atom_name.isin(allowed_atoms)]
                    fitted_bb = superimpose_apply(conf_bb, transform)
                    if not calc_global_rmsf:
                        sample_positions.append(fitted_bb[fitted_bb.res_id == res_idx + 1].coord)
                    else:
                        global_aligned.append(fitted_bb.coord)
                else:
                    if not calc_global_rmsf:
                        sample_positions.append(fitted[res_idx_in_window].coord)
                    else:
                        global_aligned.append(fitted.coord)

            if calc_global_rmsf:
                break
            else:
                aligned_positions.append(sample_positions)

        if calc_global_rmsf:
            global_aligned = np.array(global_aligned)
            global_aligned = np.transpose(global_aligned, (1, 0, 2)) if not store_full_backbone else np.transpose(global_aligned, (1, 0, 2))
            if store_full_backbone:
                assert global_aligned.shape[1] == n_draw
            aligned_positions = global_aligned

        aligned_per_ref.append(aligned_positions)
        refs_per_ref.append(ref_positions)

    progress_bar.close()

    if store_full_backbone and not calc_global_rmsf:
        zero_coords = np.zeros(3).tolist()
        for ref in range(len(aligned_per_ref)):
            refs_per_ref[ref][-1] = refs_per_ref[ref][-1].tolist()
            refs_per_ref[ref][-1].append(zero_coords)
            for i in range(len(aligned_per_ref[ref][-1])):
                aligned_per_ref[ref][-1][i] = aligned_per_ref[ref][-1][i].tolist()
                aligned_per_ref[ref][-1][i].append(zero_coords)

    refs_per_ref = np.array(refs_per_ref)
    aligned_per_ref = np.array(aligned_per_ref)

    if store_full_backbone and not calc_global_rmsf:
        aligned_per_ref = aligned_per_ref.transpose(0, 1, 3, 2, 4)
        aligned_per_ref = aligned_per_ref.reshape(aligned_per_ref.shape[:1] + (-1,) + aligned_per_ref.shape[-2:])
        aligned_per_ref = aligned_per_ref[..., :-1, :, :]
        refs_per_ref = refs_per_ref.reshape(refs_per_ref.shape[:1] + (-1,) + refs_per_ref.shape[-1:])
        refs_per_ref = refs_per_ref[..., :-1, :]

    return aligned_per_ref, refs_per_ref


def calc_rmsfs(pos: np.ndarray, ref: np.ndarray, convolution_window_size: int = 3):
    '''
    After applying get_alignment, calculate the RMSD between the aligned positions and the reference positions.
    Also apply a moving average to smooth the RMSD values.

    Parameters:
    - pos: aligned positions of shape [n_ref, N_residues, n_avg, 3]
    - ref: reference positions of shape [n_ref, N_residues, 3]
    - convolution_window_size: window size for moving average

    Returns:
    - rmsds: RMSD values of shape [n_ref, N_residues]
    '''
    rmsds_this_center = []
    for i_ref in range(pos.shape[0]):
        reference_coords = ref[i_ref]
        coords = pos[i_ref]
        rmsd = np.sqrt(np.mean(np.sum((reference_coords[:, np.newaxis] - coords)**2, axis=-1), axis=-1))
        window = np.ones(convolution_window_size) / convolution_window_size
        rmsd = np.convolve(rmsd, window, mode='same')
        rmsds_this_center.append(rmsd)
    return np.array(rmsds_this_center)


def bootstrapped_rmsf(confs:AtomArrayStack, 
                      n_bootstrap:int=10, 
                      global_alignment:bool=False, 
                      window_size:int=13,
                      n_ref:int=10, 
                      n_draw:int=100,
                      np_seed:int=123, 
                      smooth_window_size:int=1):
    
    """
    Calculate bootstrapped RMSF values for a given trajectory.
    Parameters:
    - tite: AtomArrayStack containing the trajectory conformations.
    - n_bootstrap: Number of bootstrap samples to generate.
    - global_alignment: If True, perform global alignment instead of local.
    - window_size: Size of the sliding window for local alignment. If 0 or not finite, global alignment is performed.
    - n_ref: Number of reference conformations to select for alignment.
    - n_draw: Number of conformations to randomly draw to align with each of the n_ref reference conformations.
    - np_seed: Seed for random number generation to ensure reproducibility.
    Returns:
    - mean: Mean RMSF values across bootstrap samples.
    - min_confidence: Lower bound of the 95% confidence interval.
    - max_confidence: Upper bound of the 95% confidence interval.
    """
    if not isinstance(confs, AtomArrayStack):
        raise TypeError(f"The 'tite' parameter must be an AtomArrayStack, but is {type(confs)}")
    
    rmsfs_boots = []
    for i in range(n_bootstrap):
        alignment = get_alignment(confs = confs, global_alignment=global_alignment, window_size=window_size, n_ref=n_ref, n_draw=n_draw, np_seed=np_seed+i, store_full_backbone=False)
        rmsfs = calc_rmsfs(alignment[0], alignment[1], convolution_window_size=smooth_window_size)
        rmsfs_boots.append(rmsfs)

    # shape: n_boots, n_ref, n_residues
    rmsfs_boots = np.array(rmsfs_boots)
    # take mean over n_ref conformations
    mean = np.mean(rmsfs_boots, axis=1)
    min_confidence = np.percentile(rmsfs_boots, 2.5, axis=1)
    max_confidence = np.percentile(rmsfs_boots, 97.5, axis=1)

    return mean, min_confidence, max_confidence