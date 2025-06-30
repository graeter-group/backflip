# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

from backflip.models.flexibility_module import FlexibilityModule
import torch
from omegaconf.errors import UnsupportedInterpolationType

def correct_ckpt_path(ckpt_path):
    """Correct the checkpoint path to be able to load the model: Replace paths with $now by None."""

    change_needed = False

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    for key in checkpoint['hyper_parameters']['cfg']['experiment']:
        if hasattr(checkpoint['hyper_parameters']['cfg']['experiment'][key], 'keys') and 'dirpath' in checkpoint['hyper_parameters']['cfg']['experiment'][key].keys():
            try:
                checkpoint['hyper_parameters']['cfg']['experiment'][key]['dirpath']
            except UnsupportedInterpolationType:
                checkpoint['hyper_parameters']['cfg']['experiment'][key]['dirpath'] = './'
                change_needed = True

    # Save the modified checkpoint
    if change_needed:
        torch.save(checkpoint, ckpt_path)


def get_flex_predictor(ckpt_path):
    correct_ckpt_path(ckpt_path)
    print(f'ckpt_path in get_flex_predictor: {ckpt_path}')
    flex_module = FlexibilityModule.load_from_checkpoint(checkpoint_path=ckpt_path)
    # get the model:
    flex_pred_model = flex_module.model
    flex_pred_model.eval()
    return flex_pred_model