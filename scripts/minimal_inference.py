#%%

from backflip.deployment.inference_class import BackFlip
from backflip.data.flexibility_utils import batched_rmsf_from_covar
from pathlib import Path

# path to a pdb:
rootdir = Path(__file__).parent.parent.resolve()
pdbpath = rootdir/Path('test_data/inference_examples/from_pdb_folder/1ubq.pdb')

# Load backflip model from tag:
bf = BackFlip.from_tag(tag='backflip-1.0', device='cpu')

# run prediction:
prediction = bf.predict_from_pdb(pdb_path=pdbpath)

# derive the isotropic global RMSF from the predicted per-residue covariance:
c_alpha_global_rmsf = batched_rmsf_from_covar(prediction['per_res_covariance'])[0]

print('Predicted global RMSF for C-alpha atoms:\n', c_alpha_global_rmsf)