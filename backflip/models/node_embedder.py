# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Neural network for embedding node features."""
import torch
from torch import nn
from gafl.models.utils import get_index_embedding, get_time_embedding
from omegaconf import ListConfig

class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb

        if "embed_breaks" in self._cfg:
            self.embed_breaks = self._cfg.embed_breaks
        else:
            self.embed_breaks = False
        
        if self.embed_breaks:
            self.c_breaks = 2
        else:
            self.c_breaks = 0
        
        self.c_timestep_emb = 0
        if 'c_timestep_emb' in self._cfg.keys():
            if self.c_timestep_emb == 0:
                raise DeprecationWarning("c_timestep_emb is deprecated and has no effect.")

        if "embed_aatype" in self._cfg and self._cfg.embed_aatype:
            self.embed_aatype = True
            self.aatype_embed_size = self._cfg.aatype_embed_size
            self.linear_aatype = nn.Linear(20, self.aatype_embed_size)        
        else:
            self.embed_aatype = False
            self.aatype_embed_size = 0

        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self.c_timestep_emb + self.aatype_embed_size,
            self.c_s
        )
        # node embedding layer is a composite of positional, timestep, aatype, and flexibility embeddings now;
        # hl embedding should be the resid and sequence distance 

    def forward(self, mask, aatype=None, res_idx=None, breaks=None):
        # s: [b]
        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        if res_idx is None:
            pos = torch.arange(num_res, dtype=torch.float32).to(device).unsqueeze(0)
        else:
            pos = res_idx.float()

        pos_emb = get_index_embedding(
            pos, self.c_pos_emb-self.c_breaks, max_len=2056
        )

        # [b, n_res, c_pos_emb]
        if res_idx is None:
            pos_emb = pos_emb.repeat([b, 1, 1])

        if self.embed_breaks:
            if breaks is None:
                breaks = torch.zeros([b, num_res], device=device, dtype=torch.float32)
            pos_emb = torch.cat([pos_emb, breaks.unsqueeze(-1), (1 - breaks).unsqueeze(-1)], dim=-1)
        
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [pos_emb]

        if self.embed_aatype:
            if aatype is None:
                raise ValueError("Aatype must be provided when embed_aatype is True.")
            # onehot encode:
            aatype = torch.nn.functional.one_hot(aatype, num_classes=20).float()
            aa_emb = self.linear_aatype(aatype)

            # [b, n_res, c_aatype_embed]
            aa_emb = aa_emb * mask.unsqueeze(-1)
            
            input_feats.append(aa_emb)

        return self.linear(torch.cat(input_feats, dim=-1))