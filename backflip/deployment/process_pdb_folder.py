# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import numpy as np
import os
import biotite as bt
import pandas as pd
from tqdm import tqdm
import mdtraj as md
from pathlib import Path
from backflip.deployment.utils import get_structure_tite, save_tite_as_pdb
import logging
import json
import gzip
import os
import numpy as np
import string
import argparse

class ProcessPDBs:

    '''
    Pre-process PDB files in the given folder for BackFlip inference.
    Expects monomeric proteins only (!); otherwise expects a dictionary json_chains:dict[list] to be provided
    '''
    
    def __init__(self, 
                  pdb_dir:str,
                  clean_pdb_dir:str=None,
                #   npz_dir:str=None,
                  num_processes:int=None,
                  json_chains:str=None,
                  monomers_only:bool=True, 
                  ensembles:bool=False):
        '''
        Args:
            pdb_dir: str, path to the directory containing the downloaded PDB files
            clean_pdb_dir: str, path to the directory where the clean PDB files will be saved
            npz_dir: str, path to the directory where the .npz files (processed input) will be saved
            num_processes: int, number of processes to use for multiprocessing
            json_chains: str, path to the json file containing a dict of {pdb_name:chain} pairs
        '''

        self.pdb_dir = pdb_dir
        self.clean_pdb_dir = clean_pdb_dir
        # self.npz_dir = npz_dir
        self.ensembles = ensembles

        if self.clean_pdb_dir is not None and self.npz_dir is not None:
            if not os.path.exists(clean_pdb_dir):
                os.makedirs(clean_pdb_dir)
            # if not os.path.exists(npz_dir):
            #     os.makedirs(npz_dir)
        else:
            logging.info(f'No clean_pdb_dir provided, using default paths')
            self.clean_pdb_dir = os.path.join(self.pdb_dir, 'clean_pdb')
            # self.npz_dir = os.path.join(self.pdb_dir, 'npz')
            os.makedirs(self.clean_pdb_dir, exist_ok=True)
            # os.makedirs(self.npz_dir, exist_ok=True)

        self.num_processes = num_processes
        if json_chains is not None:
            print(f'Using the provided json file with chain ids')
            with open(json_chains, 'r') as f:
                self.json_chains = json.load(f)
        else:
            self.json_chains = None
            logging.warning(f'No json file with chain ids provided! Selecting first chain from the pdb file!')
        
        self.monomers_only = monomers_only
        assert self.monomers_only == True, 'Only monomeric proteins are supported at the moment!'

    def initialize_logging(self):
        log_file = os.path.join(self.clean_pdb_dir, 'processing.log')
        logging.basicConfig(filename=log_file, 
                            filemode='a',
                            format='%(asctime)s - %(levelname)s - %(message)s', 
                            level=logging.INFO)
        logging.info('Logging initialized')
            
    def renumber_biotite(self, protein_array):
        '''
        Renumber the residues in a protein structure to be chain-wise and sequential

        Arguments:
            protein_array (AtomArray or AtomArrayStack): The protein structure
        Returns:
            AtomArray or AtomArrayStack: The renumbered protein structure
        '''
        is_stack = len(protein_array.shape) == 2  # AtomArrayStack (B, N)

        def renumber_chainwise(array):
            new_array = array.copy()
            chain_ids = np.unique(new_array.chain_id)
            sorted_chains = sorted(chain_ids)
            chain_map = {chain: letter for chain, letter in zip(sorted_chains, string.ascii_uppercase)}

            for chain in sorted_chains:
                mask = new_array.chain_id == chain
                chain_atoms = new_array[mask]
                res_ids = chain_atoms.res_id
                res_ids_new = np.unique(res_ids, return_inverse=True)[1] + 1  # Renumber residues starting from 1
                new_array.res_id[mask] = res_ids_new
                new_array.chain_id[mask] = chain_map[chain]
            return new_array

        if is_stack:
            for i in range(len(protein_array)):
                protein_array[i] = renumber_chainwise(protein_array[i])
            return protein_array
        else:
            return renumber_chainwise(protein_array)
    
    def get_modeled_seq_len(self, protein_array):
        '''
        Returns the len of CA atoms in the protein ~ modeled sequence length
        '''
        is_stack = len(protein_array.shape) == 2
        if is_stack:
            protein_array = protein_array[0]
        return len(protein_array[protein_array.atom_name == 'CA'])
        
    def has_backbone_breaks(self, protein_array, max_ca_ca_distance: float = 4.5) -> np.ndarray:
        """
        Checks for backbone breaks based on CA CA distances from a biotite AtomArrayStack or AtomArray.

        Args:
            protein_array: AtomArray or AtomArrayStack, must contain ordered backbone atoms (N, CA, C).
            max_ca_ca_distance (float): Threshold for maximum CA CA distance.

        Returns:
            np.ndarray: Boolean array of shape (batch_size,) indicating break presence per sample.
                        Returns a single bool if input is AtomArray.
        """
        coords = protein_array.coord  # (B, N, 3) or (N, 3)
        atom_names = protein_array.atom_name

        if len(protein_array.shape) == 2:  # AtomArrayStack
            ca_mask = (atom_names == 'CA')
            ca_coords = coords[:, ca_mask[0], :]  # assume consoreistent atom order across batch
            diffs = ca_coords[:, 1:, :] - ca_coords[:, :-1, :]
            dists = np.linalg.norm(diffs, axis=-1)
            return np.any(dists > max_ca_ca_distance, axis=1)
        else:  # AtomArray
            ca_coords = coords[atom_names == 'CA']
            diffs = ca_coords[1:, :] - ca_coords[:-1, :]
            dists = np.linalg.norm(diffs, axis=-1)
            return np.any(dists > max_ca_ca_distance)

    def get_sse(self, pdb_loc:str):
        '''
        Performs dssp calculation and returns the fraction of helix, strand and coil residues in the protein
        Args:
            pdb_loc: str, location of the clean pdb file containing only single selected chain
        Returns:
            sse_array: np.array [helix, strand, coil] containing the fraction of helix, strand and coil residues in the protein 
        '''
        try:
            traj = md.load(pdb_loc)
            if traj.n_frames > 1:
                traj = traj[0]
            dssp = md.compute_dssp(traj)
            dssp = dssp[0]
            dssp = dssp[np.where(dssp != 'NA')]
            helix = np.sum(dssp == 'H') / len(dssp)
            strand = np.sum(dssp == 'E') / len(dssp)
            coil = np.sum(dssp == 'C') / len(dssp)
        except:
            print(f'mdtraj failed to compute dssp on {Path(pdb_loc).name}')
            helix, strand, coil = None, None, None
        sse_array = np.array([helix, strand, coil])
        return sse_array
        
    def make_dict_bb(self, protein_array):
        '''
        Creates a dictionary to store the backbone information of the protein or stack to store as .npz file
        '''
        is_stack = len(protein_array.shape) == 2
        dict_bt = {
            'coords': protein_array.coord,
            'chain_id': protein_array.chain_id,
            'res_id': protein_array.res_id,
            'res_name': protein_array.res_name,
            'atom_name': protein_array.atom_name,
            'element': protein_array.element,
            'b_factor': np.zeros(protein_array.shape) if is_stack else np.zeros(len(protein_array)),
        }
        return dict_bt

    def write_npz(self, dict_bt:dict, pdb_name:str):
        np.savez(os.path.join(self.npz_dir, f'{pdb_name}.npz'), **dict_bt)

    def process_dataset(self):
        self.initialize_logging()

        metadata = {
            'pdb_name': [],
            'num_chains': [], 
            'num_confs':[],
            'modeled_seq_len': [],
            'helix_percent': [],
            'strand_percent': [],
            'coil_percent': [],
            'has_breaks': [],
            'processed_path': [],
        }

        pdb_files = [os.path.join(self.pdb_dir, pdb) for pdb in os.listdir(self.pdb_dir) if pdb.endswith('.pdb')]
        for pdb in tqdm(pdb_files):
            protein_name = Path(pdb).name.split('.')[0]
            try:
                protein, chains = get_structure_tite(pdb_loc=pdb, backbone=True, ensemble=self.ensembles, clean_hetero=True)
            except Exception as e:
                logging.warning(f'Could not get the protein structure for {protein_name}, skipping due to {e}')
                continue
            if self.json_chains is not None:
                chain_id = self.json_chains[protein_name]
                if chain_id in chains:
                    chains = [chain_id]  # select only the specified chain
                    protein_chain = protein[protein.chain_id == chain_id]
                else:
                    logging.warning(f'Chain {chain_id} not found in {protein_name}, skipping')
                    continue
            else:
                protein_chain = protein[protein.chain_id == chains[0]]  # select the first chain in the parsed pdb
                logging.info(f'json_chains is not provided; selecting the first chain {chains[0]} from {protein_name}')
            try:
                protein_chain = self.renumber_biotite(protein_chain)
            except Exception as e:
                logging.warning(f'Could not renumber the chain for {protein_name}, skipping due to {e}')
                continue

            save_tite_as_pdb(protein_chain, protein_name, self.clean_pdb_dir)
            modeled_seq_len = self.get_modeled_seq_len(protein_chain)
            sse = self.get_sse(os.path.join(self.pdb_dir, f'{protein_name}.pdb'))
            # dict_bb = self.make_dict_bb(protein_chain)
            # self.write_npz(dict_bb, protein_name)
            dims_chain = protein_chain.shape
            if len(dims_chain) == 2:
                num_confs = dims_chain[0]
            else:
                num_confs = 1
            metadata['pdb_name'].append(protein_name)
            metadata['num_chains'].append(len(chains))
            metadata['num_confs'].append(num_confs)
            metadata['modeled_seq_len'].append(modeled_seq_len)
            metadata['helix_percent'].append(sse[0])
            metadata['strand_percent'].append(sse[1])
            metadata['coil_percent'].append(sse[2])

            breaks_present = self.has_backbone_breaks(protein_chain)
            if np.any(breaks_present):
                metadata['has_breaks'].append(True)
            else:
                metadata['has_breaks'].append(False)

            metadata['processed_path'].append(os.path.join(self.clean_pdb_dir, f'{protein_name}.pdb'))
        logging.info('Finished processing the dataset')
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(self.clean_pdb_dir, 'metadata.csv'), index=False)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, required=True)
    parser.add_argument('--clean_pdb_dir', type=str, required=False, default=None)
    # parser.add_argument('--npz_dir', type=str, required=False, default=None)
    parser.add_argument('--json_chains', type=str, default=None)
    parser.add_argument('--ensembles', action='store_true', default=False)
    args = parser.parse_args()

    processor = ProcessPDBs(
        pdb_dir=args.pdb_dir,
        clean_pdb_dir=args.clean_pdb_dir,
        # npz_dir=args.npz_dir,
        json_chains=args.json_chains,
        monomers_only=True,
        ensembles=args.ensembles
    )
    processor.process_dataset()

if __name__ == '__main__':
    main()