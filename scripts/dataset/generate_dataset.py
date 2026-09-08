# %%
import numpy as np
import os
from pathlib import Path
from backflip.deployment.utils import chain_feats_from_pdb
import pandas as pd
from tqdm import tqdm
import torch
import mdtraj as md
from backflip.data.flexibility_utils import compute_cov_from_traj_wrt_eq, load_trajectory, dssp_traj_to_residue_fractions, DSSP_MAP, compute_dccm_from_scalar_covar, CA_cov_from_3N, load_ca_trajectory, compute_mean_cov_from_traj, per_res_cov_from_3N, compute_dssp_from_pdb
# %%

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

def _3_to_1(res_names):
    return "".join(THREE_TO_ONE.get(r, "X") for r in res_names)

def mdtraj_global_rmsf(traj_stack):
    """
    Compute per-atom RMSF relative to frame 0 using MDTraj and return Angstrom units.

    Parameters
    ----------
    traj_stack : md.Trajectory
        Trajectory stack, typically CA-only and already loaded from the target
        PDB/XTC files.

    Returns
    -------
    np.ndarray
        RMSF profile with shape ``(n_atoms,)`` in Angstrom.
    """
    if not isinstance(traj_stack, md.Trajectory):
        raise TypeError(f"traj_stack must be an mdtraj.Trajectory, got {type(traj_stack)}")
    if traj_stack.n_frames == 0:
        raise ValueError("traj_stack must contain at least one frame")

    # Match the ATLAS-style global RMSF definition by removing rigid-body
    # motion through centering and superposition onto the equilibrium frame.
    traj_stack.superpose(traj_stack, frame=0)
    return md.rmsf(traj_stack, traj_stack, frame=0) * 10.0

class AtlasDatasetGenerator:
    """
    Generates NPZ dataset files for an ATLAS-style folder containing PDB/XTC pairs.
    All required arguments are passed explicitly via __init__ and reused for every target.
    """

    def __init__(
        self,
        atlas_root: str,
        output_dir: str,
        overwrite: bool = False,
        seed: int = 123,
        covar_wrt_eq: bool = False,
    ):
        self.atlas_root = Path(atlas_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.covar_wrt_eq = covar_wrt_eq
        self.overwrite = overwrite
        np.random.seed(seed)

    def _target_seq(self, chain_feats: dict) -> str:
        ca_idx = chain_feats["atom_name"] == "CA"
        res_names = chain_feats["res_name"][ca_idx]
        return _3_to_1(res_names)

    def compute_dssp_fraction_tensors(self, pdb_path:str, xtc_paths:list):
        full_traj = load_trajectory(pdb_path, xtc_paths)
        dssp_traj = md.compute_dssp(full_traj, simplified=False)
        dssp_traj = np.array(dssp_traj)
        dssp_traj[dssp_traj == ' '] = 'L'
        dssp_fractions = dssp_traj_to_residue_fractions(dssp_traj, mapping=DSSP_MAP, loop_token="L", blank_token=" ")
        return dssp_fractions, dssp_traj

    def process_protein(self, protein_folder: Path):
        protein_name = protein_folder.name
        out_path = self.output_dir / f"{protein_name}.npz"
        if out_path.exists() and not self.overwrite:
            return

        pdb_path = protein_folder / f"{protein_name}.pdb"
        xtc_paths = sorted(protein_folder.glob(f"{protein_name}_R*.xtc"))
        chain_feats = chain_feats_from_pdb(str(pdb_path))

        seq = self._target_seq(chain_feats)
        dssp = compute_dssp_from_pdb(str(pdb_path))
        dssp_fraction, _ = self.compute_dssp_fraction_tensors(pdb_path, xtc_paths)

        traj_stack = load_ca_trajectory(str(pdb_path), xtc_paths, start_frame=0)
        global_rmsf = mdtraj_global_rmsf(traj_stack)
        if self.covar_wrt_eq:
            covariance = compute_cov_from_traj_wrt_eq(traj_stack, convert_to_angstroem=False)
        else:
            covariance = compute_mean_cov_from_traj(traj_stack, convert_to_angstroem=False)
        CA_covar = CA_cov_from_3N(torch.tensor(covariance))
        dccm_CA = compute_dccm_from_scalar_covar(CA_covar)
        per_res_covariance = per_res_cov_from_3N(torch.tensor(covariance, dtype=torch.float32))

        np.savez(
            out_path,
            res_id=chain_feats['res_id'],
            res_name=chain_feats['res_name'],
            atom_name=chain_feats['atom_name'],
            coords=chain_feats['coords'],
            chain_id=chain_feats['chain_id'],
            element=chain_feats['element'],
            global_rmsf=global_rmsf,
            dssp=dssp,
            sequence=seq,
            covariance=covariance,
            per_res_covariance=per_res_covariance,
            pairwise_couplings=CA_covar,
            pairwise_DCCM=dccm_CA,
            dssp_fraction=dssp_fraction,
        )

    def run(self):
        protein_folders = [p for p in self.atlas_root.iterdir() if p.is_dir()]
        for protein_folder in tqdm(protein_folders, desc="Generating dataset"):
            try:
                self.process_protein(protein_folder)
            except Exception as e:
                print(f"Error processing {protein_folder.name}: {e}")
                continue

# %%

if __name__ == "__main__":
    atlas_root = '<ATLAS_raw_folder>'
    output_dir = '<outdir>'
    os.makedirs(output_dir, exist_ok=True)
    generator = AtlasDatasetGenerator(
        atlas_root=atlas_root,
        output_dir=output_dir,
        overwrite=False,
        seed=123,
        covar_wrt_eq=False,
    )
    generator.run()
