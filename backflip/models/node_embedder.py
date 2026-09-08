# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Neural network for embedding node features."""
import torch
from torch import nn
from backflip.models.utils import get_index_embedding
from omegaconf import ListConfig

class NodeEmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_s = cfg.c_s
        self.c_pos_emb = cfg.c_pos_emb
        
        self.use_gating = getattr(cfg, "use_gating", False)
        self.embed_breaks = getattr(cfg, "embed_breaks", False)
        self.embed_aatype = getattr(cfg, "embed_aatype", False)
        self.aatype_embed_size = getattr(cfg, "aatype_embed_size", 0)
        
        if self.embed_aatype:
            self.linear_aatype = nn.Linear(20, self.aatype_embed_size)
            self.ln_aatype = nn.LayerNorm(self.aatype_embed_size)
            self.g_aatype = nn.Parameter(torch.tensor(2.1972))  # sigmoid ~ 0.9

        self.embed_esm = getattr(cfg, "embed_esm", False)
        self.esm_input_dim = getattr(cfg, "esm_input_dim", 0)
        self.esm_embed_size = getattr(cfg, "esm_embed_size", 0)
        self.layer_norm_esm = getattr(cfg, "layer_norm_esm", False)
        
        if self.embed_esm:
            hidden_dims = getattr(cfg, "esm_mlp_hidden_dims", None)
            if isinstance(hidden_dims, ListConfig):
                hidden_dims = list(hidden_dims)
            if not hidden_dims:
                # use a default pyramid that works for varied input sizes
                hidden_dims = [
                    max(self.esm_embed_size * 4, 512),
                    max(self.esm_embed_size * 2, 256),
                ]
            self.esm_mlp = self._build_feature_mlp(self.esm_input_dim, hidden_dims, self.esm_embed_size)
            self.ln_esm = nn.LayerNorm(self.esm_embed_size)
            self.g_esm = nn.Parameter(torch.tensor(-2.1972))  # sigmoid ~ 0.1
            self.src_dropout_p = getattr(cfg, "src_dropout_p", 0.15)

        # pos
        self.c_breaks = 2 if self.embed_breaks else 0
        self.ln_pos = nn.LayerNorm(self.c_pos_emb)
        self.g_pos = nn.Parameter(torch.tensor(0.0))  # sigmoid ~ 0.5

        # fusion MLP
        in_dim = self.c_pos_emb + (self.aatype_embed_size if self.embed_aatype else 0) + (self.esm_embed_size if self.embed_esm else 0)
        hid = max(128, int(1.5 * in_dim))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(getattr(cfg, "dropout", 0.1)),
            nn.Linear(hid, self.c_s),
        )

    def forward(self, mask, aatype=None, res_idx=None, breaks=None, esm_embedding=None):
        b, n, device = mask.shape[0], mask.shape[1], mask.device

        pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0) if res_idx is None else res_idx.float()
        pos_emb = get_index_embedding(pos, self.c_pos_emb - self.c_breaks, max_len=2056)
        if res_idx is None:
            pos_emb = pos_emb.repeat(b, 1, 1)
        if self.embed_breaks:
            if breaks is None:
                breaks = torch.zeros([b, n], device=device)
            pos_emb = torch.cat([pos_emb, breaks.unsqueeze(-1), (1 - breaks).unsqueeze(-1)], dim=-1)
        pos_emb = self.ln_pos(pos_emb) * torch.sigmoid(self.g_pos)
        pos_emb = pos_emb * mask.unsqueeze(-1)

        feats = [pos_emb]

        if self.embed_aatype:
            if aatype is None:
                raise ValueError("aatype is required")
            #NOTE: one-hot encode and linearly project including some weird stuff; num_classes=21
            aa = torch.nn.functional.one_hot(aatype, num_classes=20).float()
            aa = self.ln_aatype(self.linear_aatype(aa))
            if self.use_gating:
                aa = aa * torch.sigmoid(self.g_aatype)
            feats.append(aa * mask.unsqueeze(-1))

        if self.embed_esm:
            if esm_embedding is None:
                raise ValueError("esm_embedding is required")
            esm = self.esm_mlp(esm_embedding)
            if self.layer_norm_esm:
                esm = self.ln_esm(esm)
            if self.training and self.src_dropout_p > 0:
                if torch.rand(1, device=esm.device) < self.src_dropout_p:
                    esm = torch.zeros_like(esm)
            if self.use_gating:
                esm = esm * torch.sigmoid(self.g_esm)
            feats.append(esm * mask.unsqueeze(-1))

        z = torch.cat(feats, dim=-1)
        return self.mlp(z)

    def _build_feature_mlp(self, in_dim, hidden_dims, out_dim):
        """Create an MLP that can infer input dim when not provided."""
        layers = []
        current_dim = in_dim if in_dim and in_dim > 0 else None
        for hidden_dim in hidden_dims:
            if current_dim is None:
                layers.append(nn.LazyLinear(hidden_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)