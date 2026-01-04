#%%

from backflip.deployment.inference_class import BackFlip
from pathlib import Path

# path to a pdb:
rootdir = Path(__file__).parent.parent.resolve()
pdbpath = rootdir/Path('test_data/inference_examples/from_pdb_folder/1ubq.pdb')

# Load backflip model from tag:
bf = BackFlip.from_tag(tag='backflip-1.0', device='cpu')

# run prediction:
prediction = bf.predict_from_pdb(pdb_path=pdbpath)

c_alpha_global_rmsf = prediction['global_rmsf'][:,0] # prediction shape is (num_res, output_dim), hence the [:,0]

print('Predicted global RMSF for C-alpha atoms:\n', c_alpha_global_rmsf.cpu().numpy())