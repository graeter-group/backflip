# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import torch
import torch.nn as nn

from backflip.models.protein_embedder import ProteinEmbedderGAFL
from backflip.models.node_embedder import NodeEmbedder


def _masked_mean(x, mask, dim=1, eps=1e-8):
    mask = mask.float()
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom


class DDGPredictor(nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        self._model_conf = model_conf
        self._gfa_conf = model_conf.gfa
        self._ddg_conf = getattr(model_conf, 'ddg', None)

        self.protein_embedder = ProteinEmbedderGAFL(model_conf)

        aa_embed_dim = self._gfa_conf.c_s
        mlp_hidden = self._gfa_conf.c_s
        if self._ddg_conf is not None:
            if hasattr(self._ddg_conf, 'aa_embed_dim'):
                aa_embed_dim = int(self._ddg_conf.aa_embed_dim)
            if hasattr(self._ddg_conf, 'mlp_hidden'):
                mlp_hidden = int(self._ddg_conf.mlp_hidden)

        self.aa_embed = nn.Embedding(21, aa_embed_dim, padding_idx=20)
        self.readout = nn.Sequential(
            nn.LayerNorm(self._gfa_conf.c_s + 2 * aa_embed_dim),
            nn.Linear(self._gfa_conf.c_s + 2 * aa_embed_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, batch):
        node_embed, _ = self.protein_embedder(batch)
        mut_mask = batch['mut_mask']

        h_mut = _masked_mean(node_embed, mut_mask, dim=1)

        wt_aa = batch['wt_aa']
        mut_aa = batch['mut_aa']
        mut_pos_mask = batch['mut_pos_mask']

        wt_aa = wt_aa.clone()
        mut_aa = mut_aa.clone()
        wt_aa[wt_aa < 0] = 20
        mut_aa[mut_aa < 0] = 20

        wt_emb = self.aa_embed(wt_aa)
        mut_emb = self.aa_embed(mut_aa)
        aa_feat = torch.cat([wt_emb, mut_emb], dim=-1)
        aa_feat = _masked_mean(aa_feat, mut_pos_mask, dim=1)

        h = torch.cat([h_mut, aa_feat], dim=-1)
        ddg_pred = self.readout(h).squeeze(-1)
        return ddg_pred


class NoIPAMLP(nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        self._model_conf = model_conf
        self._gfa_conf = model_conf.gfa
        self._ddg_conf = getattr(model_conf, 'ddg', None)

        aa_embed_dim = self._gfa_conf.c_s
        mlp_hidden = self._gfa_conf.c_s
        if self._ddg_conf is not None:
            if hasattr(self._ddg_conf, 'aa_embed_dim'):
                aa_embed_dim = int(self._ddg_conf.aa_embed_dim)
            if hasattr(self._ddg_conf, 'mlp_hidden'):
                mlp_hidden = int(self._ddg_conf.mlp_hidden)

        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.aa_embed = nn.Embedding(21, aa_embed_dim, padding_idx=20)
        self.readout = nn.Sequential(
            nn.LayerNorm(self._gfa_conf.c_s + 2 * aa_embed_dim),
            nn.Linear(self._gfa_conf.c_s + 2 * aa_embed_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, batch):
        node_mask = batch['res_mask']
        res_idx = batch['res_idx'] if self._model_conf.node_features.embed_res_idx else None
        aatype = batch['aatype'] if self._model_conf.node_features.embed_aatype else None
        node_embed = self.node_embedder(node_mask, res_idx=res_idx, aatype=aatype)
        node_embed = node_embed * node_mask.unsqueeze(-1)

        mut_mask = batch['mut_mask']
        h_mut = _masked_mean(node_embed, mut_mask, dim=1)

        wt_aa = batch['wt_aa'].clone()
        mut_aa = batch['mut_aa'].clone()
        mut_pos_mask = batch['mut_pos_mask']
        wt_aa[wt_aa < 0] = 20
        mut_aa[mut_aa < 0] = 20

        wt_emb = self.aa_embed(wt_aa)
        mut_emb = self.aa_embed(mut_aa)
        aa_feat = torch.cat([wt_emb, mut_emb], dim=-1)
        aa_feat = _masked_mean(aa_feat, mut_pos_mask, dim=1)

        h = torch.cat([h_mut, aa_feat], dim=-1)
        ddg_pred = self.readout(h).squeeze(-1)
        return ddg_pred
