# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import biotite.structure.io as strucio
from biotite.structure import atoms as atoms_tite
from biotite.structure.superimpose import superimpose, superimpose_apply
from biotite.structure import AtomArrayStack
import numpy as np
from tqdm import tqdm

import biotite.structure.io.pdb as pdb
from pathlib import Path

def get_alignment(confs: atoms_tite.AtomArrayStack,
                  global_alignment: bool = False,
                  window_size: int = 7,
                  n_ref: int = 10,
                  n_draw: int = 10,
                  np_seed=123,
                  store_full_backbone: bool = False):
    """
    Align the positions of alpha carbons (CA atoms) in a trajectory to a set of reference conformations.

    Arguments:
    - confs: AtomArrayStack containing the trajectory conformations.
    - global_alignment: If True, align the whole protein instead of a sliding window.
    - window_size: Size of the sliding window for local alignment. If 0 or not finite, global alignment is performed.
    - n_ref: Number of reference conformations to select for alignment.
    - n_draw: Number of conformations to randomly draw to aling with each of the n_ref reference conformations.
    - np_seed: Seed for random number generation to ensure reproducibility.
    - store_full_backbone: If True, store the full backbone atoms (N, CA, C, O) instead of just CA.
    
    Returns:
    - aligned_positions: np.ndarray(float) of shape (n_ref, N_residues, n_draw, 3)
    - ref_positions: np.ndarray(float) of shape (n_ref, N_residues, 3)
    """
    np.random.seed(np_seed)
    if not global_aligned and window_size <= 0:
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

    total_calls = n_ref if calc_global_rmsf else num_res * n_ref
    progress_bar = tqdm(total=total_calls, desc=f"Aligning frames")

    num_frames = len(confs)
    print(f"Number of frames: {num_frames}")

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

def calc_rmsds(pos: np.ndarray, ref: np.ndarray, convolution_window_size: int = 3):
    '''
    After applying get_alignment, calculate the RMSD between the aligned positions and the reference positions.
    Also apply a moving average to smooth the RMSD values.

    Args:
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
