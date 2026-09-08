#%%

from backflip.deployment.inference_class import BackFlip
from backflip.data.flexibility_utils import batched_rmsf_from_covar
from pathlib import Path
import matplotlib.pyplot as plt

rootdir = Path(__file__).parent.parent.resolve()
#%%

######################################################################
# 0: PREDICT PER-RESIDUE COVARIANCE FROM A SINGLE PDB FILE
######################################################################

pdbpath = rootdir/Path('test_data/inference_examples/from_pdb_folder/1ubq.pdb')

# Load backflip model from tag (downloads the checkpoint on first use):
model = BackFlip.from_tag(tag='backflip-2.1', device='cpu')
prediction = model.predict_from_pdb(pdb_path=pdbpath)
#%%

# BackFlip predicts three flexibility features:

#   - per_res_covariance: (N, 3, 3) anisotropic covariance of each residue's own fluctuations.
#     Its trace gives the (isotropic) mean squared fluctuation, so
#     sqrt(trace(per_res_covariance)) recovers global RMSF profile (see the paper for more details)
#     batched_rmsf_from_covar does exactly this, batched over residues (and proteins).

#   - pairwise_couplings: (N, N) raw CA-CA covariance between residue pairs.

#   - pairwise_DCCM: (N, N) the normalized version of pairwise_couplings (values in [-1, 1]),
#     the dynamic cross-correlation matrix. Entries close to +1 mean the two residues move
#     in a correlated fashion (same direction), entries close to -1 mean they move in an
#     anticorrelated fashion (opposite directions).

per_res_covariance = prediction['per_res_covariance']
pairwise_dccm = prediction['pairwise_DCCM']

global_rmsf = batched_rmsf_from_covar(per_res_covariance)[0]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

# Plot the RMSF profile derived from per_res_covariance
ax[0].plot(global_rmsf, label='Global RMSF', linewidth=3.0)
ax[0].set_xlabel('Residue index', fontsize=22)
ax[0].set_ylabel(r'Global RMSF [$\AA$]', fontsize=24)
ax[0].tick_params(labelsize=22)

# Plot pairwise_DCCM as a correlated/anticorrelated residue-motion heatmap
im = ax[1].imshow(pairwise_dccm, cmap='RdBu_r', vmin=-1, vmax=1)
ax[1].set_xlabel('Residue index', fontsize=22)
ax[1].set_ylabel('Residue index', fontsize=22)
ax[1].tick_params(labelsize=22)
fig.colorbar(im, ax=ax[1], label='DCCM')

fig.suptitle('BackFlip Flexibility Prediction for 1UBQ', fontsize=24)
plt.tight_layout()
plt.savefig(rootdir/Path('1ubq_backflip_flexibility_prediction.png'), dpi=300)
plt.close()

#%%

######################################################################
# 1: ANNOTATE PDB DATASETS WITH ISOTROPIC RMSF AS A B-FACTOR
######################################################################

# Inference on the folder containing .pdb files.
pdb_folder_test = rootdir/Path('test_data/inference_examples/from_pdb_folder')

# Load backflip model from tag:
bf = BackFlip.from_tag(tag='backflip-2.1', device='cpu', progress_bar=True) # change device to gpu if available

# Predict and write results to output_folder (a folder called 'inference_results' next to the
# input files, since output_folder=None and overwrite=False). Two subfolders are created:
#   - npz/   : always written, one *_pred.npz file per input with the raw
#              per_res_covariance/pairwise_couplings/pairwise_DCCM predictions.
#   - cifs/  : only written if rmsf_as_bfactor=True, one .cif file per input with the
#              isotropic RMSF (derived from per_res_covariance) in the B-factor column.
# If overwrite=True instead, results are written next to (or over) the input files directly.
bf.predict(input_path=pdb_folder_test,
           output_folder=None,
           overwrite=False,
           cuda_memory_GB=8,
           rmsf_as_bfactor=True)
#%%

# Visualize results from the B-factors of the prediction. Assuming the inference was run with overwrite=False as above

from backflip.deployment.utils import profile_from_bfac

# rmsf_as_bfactor=True writes a .cif file (not .pdb) with the isotropic RMSF in the B-factor
# column, into the 'cifs' subfolder of the output folder:
inference_loc_test = pdb_folder_test / 'inference_results' / 'cifs' / '5pc9.cif'
assert Path(inference_loc_test).exists(), f'Inference results not found at {inference_loc_test}! First run inference as explained above.'

# Loading global_rmsf profile for CA atoms from B-factors for 5pc9:
global_rmsf_CA = profile_from_bfac(inference_loc_test)

# Plot global_rmsf profile
plt.plot(global_rmsf_CA, label='global_rmsf', linewidth=2.0)
plt.xlabel('Residue index', fontsize=16)
plt.ylabel(r'Predicted global RMSF [$\AA$]', fontsize=16)
plt.tick_params(labelsize=14)
plt.title('BackFlip Global RMSF Prediction for 5PC9', fontsize=18)
plt.tight_layout()
plt.savefig(rootdir/Path('scripts')/Path('5pc9_backflip_global_rmsf.png'), dpi=300)

#%%

#######################################################################
# 2: LOAD FRAME REPRESENTATION FROM A PDB FILE AND RUN INFERENCE DIRECTLY ON THAT
#######################################################################

# If you want to combine BackFlip with another model that uses the SE(3)^N frame representation of proteins, you can also call it directly on that representation.
# This can be especially useful if you want to guide generative models with BackFlip, for example as in the 'Flexibility-Conditioned Protein Structure Design with Flow Matching' paper.

from backflip.deployment.utils import frames_from_pdb_tite

pdb_loc_test = pdb_folder_test / '5pca.pdb'

# Input to the model is a set of translations and rotations
model_input, seq = frames_from_pdb_tite(pdb_loc_test)

# Putting inputs in a list is needed to define the batch dimension if inference is done on batches:
translations = [model_input['trans_1']]
rotations = [model_input['rotmats_1']]
aatype = [model_input['aatype']]

prediction = bf.predict_from_frames(translations=translations, rotations=rotations, cuda_memory_GB=8, aatypes=aatype)

# output is a list with batch (B, ) dimension where each idx corresponds to the input sample:
per_res_covariance = prediction[0]['per_res_covariance']
global_rmsf = batched_rmsf_from_covar(per_res_covariance)[0]
print(f'Global RMSF shape: {global_rmsf.shape}')
# %%

# For application within downstream models, pass the frames directly as dict (here shapes (batchdim, n_res,) is needed), skipping batchsize calculation and consistency checks

import torch

num_res = model_input['trans_1'].shape[0]
device = 'cpu'

batch = {
    'rotmats_1': model_input['rotmats_1'].unsqueeze(0),  # add batch dimension
    'trans_1': model_input['trans_1'].unsqueeze(0),  # add batch dimension
    'res_idx': torch.arange(num_res).unsqueeze(0),
    'aatype': model_input['aatype'].unsqueeze(0),
    'res_mask': torch.ones_like(model_input['trans_1'][..., 0]).unsqueeze(0)
}
batch = {k: v.to(device) for k, v in batch.items()}  # move to device

# bf(batch) returns the raw model output {'node': {}, 'edge': {...}}, still batched:
per_res_covariance = bf(batch)['edge']['per_res_covariance'].detach().cpu().numpy()  # shape (B, N, 3, 3)
global_rmsf = batched_rmsf_from_covar(per_res_covariance)[0]
print(f'Global RMSF shape: {global_rmsf.shape}')