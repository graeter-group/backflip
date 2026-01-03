# %%

'''This script is an example of how to compute local flexibility from an ensemble as bootstrapped local RMSF'''

from backflip.data.flexibility_utils import biotite_from_pdb, compute_rmsf, load_ca_trajectory
import mdtraj as md
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# %%

dummy_traj = Path('../test_data/rmsf_examples/3a5e.pdb').resolve()
tite_traj = biotite_from_pdb(dummy_traj)

# %%

############################################################
# 1: LOCAL RMSF CALCULATION
############################################################

# Compute the local structural alignment and the RMSF of the trajectory with the alignment window size of 13 residues.
# We do not pick a single conformation as reference for rmsf but calculate the mean among 10 randomly selected reference conformations.
# This is quite expensive to compute since the local alignment of 13 residues is carried out (n_res, n_ref) times
N_RESIDUES_IN_WINDOW = 13
N_REFERENCE_CONFORMATIONS = 10
local_rmsf = compute_rmsf(tite_traj, window_size=N_RESIDUES_IN_WINDOW, n_ref=N_REFERENCE_CONFORMATIONS)
# %%

############################################################
# 2: Visualization
############################################################

# Visualize per-residue local RMSF
plt.plot(local_rmsf.mean(axis=0), label='mean')
# Plot the standard deviation across 10 n_ref conformations
plt.fill_between(range(len(local_rmsf.mean(axis=0))), 
                 local_rmsf.mean(axis=0) - local_rmsf.std(axis=0), 
                 local_rmsf.mean(axis=0) + local_rmsf.std(axis=0), 
                 alpha=0.2, label='stddev')
plt.xlabel('Residue index')
plt.ylabel('Local RMSF')
plt.title('Local RMSF of 3a5e')
# %%

############################################################
# 3: COMPARISON WITH GLOBAL RMSF
# Defined as the RMSF after alignment of the whole ensemble onto the equilibrium state; as deposited in ATLAS.
############################################################

# Load CA trajectory
ca_traj = load_ca_trajectory(dummy_traj)
superposed_traj = ca_traj.superpose(ca_traj, 0)
global_rmsf = md.rmsf(superposed_traj, superposed_traj, 0)*10 # convert to Angstrom

# %%

# Visualize the global RMSF as in ATLAS
plt.plot(global_rmsf, label='mdtraj rmsf')
plt.xlabel('Residue index')
plt.ylabel('Global RMSF [$\AA$]')
plt.title('Global RMSF of 3a5e as in ATLAS')