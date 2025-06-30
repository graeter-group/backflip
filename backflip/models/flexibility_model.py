# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import torch
from backflip.models.protein_embedder import ProteinEmbedderGAFL
import torch.nn as nn

class FlexibilityModel(nn.Module):
        
    def __init__(self, model_conf):
        super().__init__()

        self.protein_embedder = ProteinEmbedderGAFL(model_conf)

        self._gfa_conf = model_conf.gfa
        self._flexibility_conf = model_conf.flexibility

        self.outputs = self._flexibility_conf.outputs
        self.out_dims = self._flexibility_conf.out_dims
        self.max_values = [float(max_value) if max_value is not None else None for max_value in self._flexibility_conf.max_values]

        assert len(self.outputs) == len(self.out_dims), f"Number of outputs and output dimensions must match. Got {len(self.outputs)} outputs and {len(self.out_dims)} output dimensions."

        assert len(self.outputs) == len(self.max_values), f"Number of outputs and max values must match. Got {len(self.outputs)} outputs and {len(self.max_values)} max values. Set entries to null if no max value is needed."

        self.output_dim = sum(self._flexibility_conf.out_dims)


        self.output_mlp = nn.Sequential(
            nn.LayerNorm(self._gfa_conf.c_s),
            nn.Linear(self._gfa_conf.c_s, self._gfa_conf.c_s),
            nn.ELU(),
            # nn.Linear(self._gfa_conf.c_s, self._gfa_conf.c_s), # for global rmsf settings!
            # nn.ELU(), # for global rmsf settings!
            nn.Linear(self._gfa_conf.c_s, self.output_dim)
            )


    def forward(self, input_feats):
        """"
        Output is a dictionary of {output_name: output_tensor} with output_tensor of shape (*batch_shape, num_res, out_dim[out_name])
        """
        node_feats, _ = self.protein_embedder(input_feats)
        node_feats = self.output_mlp(node_feats)

        output = {}

        start_dim = 0

        for i, output_name in enumerate(self._flexibility_conf.outputs):
            output[output_name] = node_feats[..., start_dim:start_dim+self.out_dims[i]]
            start_dim += self.out_dims[i]

            if self.max_values[i] is not None:
                output[output_name] = torch.sigmoid(output[output_name]) * self.max_values[i]

        return output