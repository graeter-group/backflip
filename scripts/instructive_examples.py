#%%

from backflip.deployment.inference_class import BackFlip
from pathlib import Path
import matplotlib.pyplot as plt

rootdir = Path(__file__).parent.parent.resolve()
#%%

######################################################################
# 0: PREDICT GLOBAL/LOCAL FLEXIBILITY FROM A SINGLE PDB FILE
######################################################################

pdbpath = rootdir/Path('test_data/inference_examples/from_pdb_folder/1ubq.pdb')

# Load backflip model trained without sequence embedding from tag:
bf = BackFlip.from_tag(tag='backflip-1.0', device='cpu')
prediction = bf.predict_from_pdb(pdb_path=pdbpath)
#%%

local_flex = prediction['local_flex']
global_rmsf = prediction['global_rmsf']

fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))

# Plot local_flex profile
ax[0].plot(local_flex, label='Local Flexibility', linewidth=3.0)
ax[0].set_xlabel('Residue index', fontsize=22)
ax[0].set_ylabel(r'Local Flexibility [$\AA$]', fontsize=24)
ax[0].tick_params(labelsize=22)

# Plot global_rmsf profile
ax[1].plot(global_rmsf, label='Global RMSF', linewidth=3.0)
ax[1].set_xlabel('Residue index', fontsize=22)
ax[1].set_ylabel(r'Global RMSF [$\AA$]', fontsize=24)
ax[1].tick_params(labelsize=22)

fig.suptitle('BackFlip Local/Global Flexibility Prediction for 1UBQ', fontsize=24)
plt.tight_layout()
plt.savefig(rootdir/Path('1ubq_backflip_flexibility_prediction.png'), dpi=300)
plt.close()
#%%

# While the original backflip model predicts flexibility from structure alone, we also provide a second model that uses amino acid information to slightly improve accuracy, if a sequence is available:
bf_seq = BackFlip.from_tag(tag='backflip-1.0-seq', device='cpu')
prediction_seq = bf_seq.predict_from_pdb(pdb_path=pdbpath)
global_rmsf_seq = prediction_seq['global_rmsf']

# compare the two global rmsf predictions
plt.figure(figsize=(8,6))
plt.plot(global_rmsf, label='Global RMSF predicted without sequence', linewidth=2.0)
plt.plot(global_rmsf_seq, label='Global RMSF predicted with sequence', linewidth=2.0)
plt.xlabel('Residue index', fontsize=16)
plt.ylabel(r'Predicted global RMSF [$\AA$]', fontsize=16)
plt.tick_params(labelsize=14)
plt.title('BackFlip Global RMSF Prediction for 1UBQ', fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(rootdir/Path('scripts')/Path('1ubq_backflip_global_rmsf_comparison.png'), dpi=300)
plt.close()

#%%

######################################################################
# 1: ANNOTATE PDB DATASETS WITH GLOBAL/LOCAL FLEXIBILITY
######################################################################

# Inference on the folder containing .pdb files.
pdb_folder_test = rootdir/Path('test_data/inference_examples/from_pdb_folder')

# Load backflip model from tag:
bf = BackFlip.from_tag(tag='backflip-1.0', device='cpu', progress_bar=True) # change device to gpu if available

# Configure whether to output global_rmsf or local_flex:
bf.rmsf_type = 'global_rmsf' # options are 'global_rmsf' or 'local_flex'

# Predict and write local RMSF as a b-factor to the pdb files. If overwrite is set to True, the b-factor will be written to the original pdb files. Else will write new .pdb files to the output_folder.
# If output_folder is None, will write to a folder called 'inference_results' in the input pdb folder.
bf.predict(input_path=pdb_folder_test,
           output_folder=None,
           overwrite=False,
           cuda_memory_GB=8)

#%%

# Visualize results from the B-factors of the prediction. Assuming the inference was run with overwrite=False as above

from backflip.deployment.utils import profile_from_bfac

# Assuming we wrote global_rmsf as a b-factor in the pdb files:
inference_loc_test = pdb_folder_test / 'inference_results' / '5pc9.pdb'
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

from backflip.deployment.utils import frames_from_pdb

pdb_loc_test = pdb_folder_test / '5pca.pdb'

# Input to the model is a set of translations and rotations
model_input = frames_from_pdb(pdb_loc_test)

# Putting inputs in a list is needed to define the batch dimension if inference is done on batches:
translations = [model_input['trans_1']]
rotations = [model_input['rotmats_1']]

prediction = bf.predict_from_frames(translations=translations, rotations=rotations, cuda_memory_GB=8)

# output is a list with batch (B, ) dimension where each idx corresponds to the input sample:
global_rmsf = prediction[0]['global_rmsf']
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
    'res_mask': torch.ones_like(model_input['trans_1'][..., 0]).unsqueeze(0)
}
batch = {k: v.to(device) for k, v in batch.items()}  # move to device
global_rmsf = bf(batch)['global_rmsf'][0]
print(f'Global RMSF shape: {global_rmsf.shape}')
