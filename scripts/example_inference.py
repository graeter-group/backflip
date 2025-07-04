#%%

from backflip.deployment.inference_class import BackFlip
from pathlib import Path
import matplotlib.pyplot as plt

rootdir = Path(__file__).parent.parent.resolve()
#%%

######################################################################
# 0: PREDICT FLEXIBILITY FROM A PDB FILE
######################################################################

pdbpath = rootdir/Path('test_data/inference_examples/from_pdb_folder/5pc9.pdb')

# Load backflip model from tag:
bf = BackFlip.from_tag(tag='backflip-0.2', device='cpu')

prediction = bf.predict_from_pdb(pdb_path=pdbpath)

#%%

local_flex = prediction['local_flex']
global_rmsf = prediction['global_rmsf']

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot local_flex profile
ax[0].plot(local_flex, label='local_flex', linewidth=2.0)
ax[0].set_xlabel('Residue index', fontsize=16)
ax[0].set_ylabel('Local Flexibility [$\AA$]', fontsize=16)
ax[0].tick_params(labelsize=14)

# Plot global_rmsf profile
ax[1].plot(global_rmsf, label='global_rmsf', linewidth=2.0)
ax[1].set_xlabel('Residue index', fontsize=16)
ax[1].set_ylabel('Global RMSF [$\AA$]', fontsize=16)
ax[1].tick_params(labelsize=14)

fig.suptitle('BackFlip Local/Global Flexibility Prediction', fontsize=18)
plt.tight_layout()

# # Plot local_flex profile
# plt.plot(local_flex, label='local_flex', linewidth=2.0)
# plt.xlabel('Residue index', fontsize=16)
# plt.ylabel('Predicted Local Flexibility [$\AA$]', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.show()

# # Plot global_rmsf profile
# plt.plot(global_rmsf, label='global_rmsf', linewidth=2.0)
# plt.xlabel('Residue index', fontsize=16)
# plt.ylabel('Predicted Global RMSF [$\AA$]', fontsize=16)
# plt.tick_params(labelsize=14)
# plt.show()


#%%

######################################################################
# 1: ANNOTATE FLEXIBILITY AS B FACTOR IN PDB FILES
######################################################################

# Inference on the folder containing .pdb files.
pdb_folder_test = rootdir/Path('test_data/inference_examples/from_pdb_folder')

# Load backflip model from tag:
bf = BackFlip.from_tag(tag='backflip-0.2', device='cuda', progress_bar=True)

# Predict and write local RMSF as a b-factor to the pdb files. If overwrite is set to True, the b-factor will be written to the original pdb files. Else will write new .pdb files to a new folder called inference_results.
bf.predict(input_path=pdb_folder_test,
           output_folder=None,
           overwrite=False,
           cuda_memory_GB=8)

#%%

# Visualize results from the B-factors of the prediction. Assuming the inference was run with overwrite=False as above

from backflip.deployment.utils import profile_from_bfac

inference_loc_test = pdb_folder_test / 'inference_results' / '5pc9.pdb'

assert Path(inference_loc_test).exists(), f'Inference results not found at {inference_loc_test}! First run inference as explained above.'

# loading local_flex profile for CA atoms from B-factors for 5pc9:
local_flex_CA = profile_from_bfac(inference_loc_test)

# Plot local_flex profile
plt.plot(local_flex_CA, label='local_flex', linewidth=2.0)
plt.xlabel('Residue index', fontsize=16)
plt.ylabel('Predicted Local Flexibility [$\AA$]', fontsize=16)
plt.tick_params(labelsize=14)
plt.show()

#%%

#######################################################################
# 2: LOAD FRAME REPRESENTATION FROM A PDB FILE AND RUN INFERENCE
#######################################################################

from backflip.deployment.utils import frames_from_pdb

pdb_loc_test = pdb_folder_test / '5pca.pdb'

# Input to the model is a set of translations and rotations
model_input = frames_from_pdb(pdb_loc_test)

# Putting inputs in a list is needed to define the batch dimension if inference is done on batches:
translations = [model_input['trans_1']]
rotations = [model_input['rotmats_1']]

prediction = bf.predict_from_frames(translations=translations, rotations=rotations,cuda_memory_GB=8)

# output is a list with batch (B, ) dimension where each idx corresponds to the input sample:
local_flex = prediction[0]['local_flex']

print(f'Local flexibility shape: {local_flex.shape}')
# %%


# For application within downstream models, pass the frames directly as dict (here shapes (batchdim, n_res,) is needed), skipping batchsize calculation and consistency checks

import torch

num_res = model_input['trans_1'].shape[0]
device = 'cuda'

batch = {
    'rotmats_1': model_input['rotmats_1'].unsqueeze(0),  # add batch dimension
    'trans_1': model_input['trans_1'].unsqueeze(0),  # add batch dimension
    'res_idx': torch.arange(num_res).unsqueeze(0),
    'res_mask': torch.ones_like(model_input['trans_1'][..., 0]).unsqueeze(0)
}
batch = {k: v.to(device) for k, v in batch.items()}  # move to device
local_flex = bf(batch)['local_flex'][0]

print(f'Local flexibility shape: {local_flex.shape}')
# %%
