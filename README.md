# BackFlip - Backbone Flexibility Predictor

![BackFlip](assets/backflip_github_small.png)

## Description

BackFlip is a model trained to predict **per-residue backbone flexibility** of protein structures described in the paper [Flexibility-conditioned protein structure design with flow matching](https://openreview.net/forum?id=890gHX7ieS).

This repository relies on the [GAFL](https://github.com/hits-mli/gafl) package and code from [FrameFlow](https://github.com/microsoft/protein-frame-flow).

---

## Inference

Inference on the example folder containing .pdb files:

```python
from backflip.deployment.inference_class import BackFlip
from pathlib import Path

# Inference on the folder containing .pdb files.
pdb_folder_test = Path('./test_data/inference_examples/from_pdb_folder').resolve()

# Download model weights and load backflip model from tag:
bf = BackFlip.from_tag(tag='backflip-0.2', device='cuda', progress_bar=True)

# Predict and write local RMSF as a b-factor to the pdb files
bf.predict(pdb_folder=pdb_folder_test,cuda_memory_GB=8)
```

We recommend running inference with BackFlip given a folder containing .pdb or .cif files as input. You can also point just to the structural file itself. For more details and brief analyses we refer to the example inference scripts available at `scripts/example_inference.py`.

### Pre-process input structures for inference

BackFlip default inference script expects clean .pdb or .cif **monomeric** structural files as input **without breaks**. We provide pre-processing script that cleans PDBs. Run it on the folder containing structural files with:

```python
python scripts/process_pdb_folder.py --pdb_dir <path>
```

This will create a folder `clean_pdb` with cleaned PDBs along with a `metadata.csv` with some information on the protein structures. BackFlip is not (yet) robust to structural breaks, thus we discourage running inference on protein sturctures which have `has_breaks == True` in the `metadata.csv`. This can cause artifacts and unrealistic predictions!

### Compute local or global RMSF

We provide a detailed explanation on how to compute local or global RMSF in `scripts/example_rmsf.py`.

---

## Installation

## Installation script

You can use our install script (here for torch version 2.6.0 and cuda 12.4), which esssentially executes the steps specified in the section **pip** below:

```bash
git clone https://github.com/graeter-group/backflip.git
conda create -n backflip python=3.10 pip=23.2.1 -y
conda activate backflip && bash backflip/install_utils/install_via_pip.sh 2.6.0 124 #torch-ver and cuda as args
```

Verify your installation by running our example script:

```bash
cd backflip/ && python backflip/scripts/example_inference.py
```

## pip

Optional: Create a virtual environment, e.g. with conda, and install pip23.2.1:

```bash
conda create -n backflip python=3.10 pip=23.2.1 -y
conda activate backflip
```

Install the dependencies from the requirements file:

```bash
git clone https://github.com/graeter-group/backflip.git
pip install -r backflip/install_utils/requirements.txt

# BackFlip builds on top of the GAFL package, which is installed from source:
git clone https://github.com/hits-mli/gafl.git
cd gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e . # Install GAFL
cd ..

# Finally, install backflip with pip:
cd backflip
pip install -e .
```

Install torch with a suitable cuda version, e.g.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

where you can replace cu124 by your cuda version, e.g. cu118 or cu121.

## conda

FliPS relies on the [GAFL](https://github.com/hits-mli/gafl) package, which can be installed from GitHub as shown below. The dependencies besides GAFL are listed in `install_utils/environment.yaml`, we also provide a minimal environment in `install_utils/minimal_env.yaml`, where it is easier to change torch/cuda versions.

```bash
# download backflip:
git clone https://github.com/graeter-group/backflip.git
# create env with dependencies:
conda env create -f backflip/install_utils/minimal_env.yaml
conda activate backflip

# install gafl:
git clone https://github.com/hits-mli/gafl.git
cd gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e .
cd ..

# install backflip:
cd backflip
pip install -e .
```

## Common installation issues

Problems with torch_scatter can usually be resolved by uninstalling and re-installing it via pip for the correct torch and cuda version, e.g. `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu124.html` for torch 2.0.0 and cuda 12.4.

## Citation

```
@inproceedings{
viliuga2025flexibilityconditioned,
title={Flexibility-conditioned protein structure design with flow matching},
author={Vsevolod Viliuga and Leif Seute and Nicolas Wolf and Simon Wagner and Arne Elofsson and Jan St{\"u}hmer and Frauke Gr{\"a}ter},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=890gHX7ieS}
}
```

The code relies on the [GAFL](https://github.com/hits-mli/gafl) package and code from [FrameFlow](https://github.com/microsoft/protein-frame-flow). It would be appreciated if you also cite the two respective papers if you use the code.