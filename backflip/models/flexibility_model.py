# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import torch
from backflip.models.protein_embedder import ProteinEmbedderIPA
from backflip.data.flexibility_utils import compute_dccm_from_scalar_covar
import torch.nn as nn
import torch.nn.functional as F

class FlexibilityModelIPA(nn.Module):
        
    def __init__(self, model_conf):
        super().__init__()
    
        self.protein_embedder = ProteinEmbedderIPA(model_conf)
        self._ipa_conf = model_conf.ipa
        self._flexibility_conf = model_conf.flexibility
        self._edge_feats_conf = model_conf.edge_features

        self.node_outs = self._flexibility_conf.node_outputs
        self.edge_outs = self._flexibility_conf.edge_outputs
        
        self.edge_embed_dim = self._ipa_conf.c_z
        self.node_embed_dim = self._ipa_conf.c_s
        
        # per-residue N,3,3 head config:
        self.equivariant_covar = self._edge_feats_conf.get('equivariant_covariance', True)
        self.symmetrize_edges_ca = self._edge_feats_conf.get('symmetrize_edges_CA', False)
        self.fuse_node_edges_per_res_cov = self._edge_feats_conf.get('fuse_node_edges_per_res_cov', False)
        self.ablate_cholesky_per_res_head = self._edge_feats_conf.get('ablate_cholesky_per_res_head', False)
        self.use_only_nodes_per_res_cov = self._edge_feats_conf.get('use_only_nodes_per_res_cov', False)

        self.per_res_cov_head = per_res_cov_head(
            node_dim=self.node_embed_dim,
            ablate_cholesky=self.ablate_cholesky_per_res_head,
        )

        # N,N pairwise covariance head config:
        self.fuse_node_edges_ca_cov = self._edge_feats_conf.get('fuse_node_edges_ca_cov', False)
        self.ablate_cholesky_CA_head = self._edge_feats_conf.get('ablate_cholesky_CA_head', False)

        self.pairwise_coupling_head = pairwise_coupling_head(
            edge_dim=self.edge_embed_dim,
            symmetrize_edges=self.symmetrize_edges_ca,
            ablate_cholesky=self.ablate_cholesky_CA_head,
		)

    def forward(self, input_feats):
        """"
        Output is a dictionary of {output_name: output_tensor} with output_tensor of shape (*batch_shape, num_res, out_dim[out_name])
        """
        node_embeds, edge_embeds = self.protein_embedder(input_feats)
        if self.equivariant_covar:
            assert 'rotmats_1' in input_feats, "FlexibilityModelIPA needs rotmats_1 in input_feats"
            rotmats = input_feats['rotmats_1']  #(B, N, 3, 3)
    
        output = {"node": {}, "edge": {}}
        cov_CA, dccm_CA = self.pairwise_coupling_head(edge_embeds)  #SPD covariance (B, N, N) and DCCM (B, N, N)
        per_res_cov = self.per_res_cov_head(node_embeds)  #(B, N, 3, 3)

        if self.equivariant_covar:
            # rotate each block to global frame
            B, N, _, _ = per_res_cov.shape
            R_t = rotmats.transpose(-1, -2)
            per_res_cov = rotmats @ per_res_cov @ R_t
        
        output["edge"]["per_res_covariance"] = per_res_cov
        output["edge"]["pairwise_couplings"] = cov_CA
        output["edge"]["pairwise_DCCM"] = dccm_CA
        return output

class per_res_cov_head(nn.Module):
    """
    Predict per-residue 3x3 covariance blocks from node features only.

    node_embeds: (B, N, D)

    Output:
        cov_blocks: (B, N, 3, 3) with cov = A@A^T
    """
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int | None = None,
        eps: float = 1e-4,
        ablate_cholesky: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.ablate_cholesky_per_res_head = ablate_cholesky
        
        self.input_dim = node_dim if hidden_dim is None else hidden_dim
        self.node_proj = (
            nn.Identity()
            if self.input_dim == node_dim
            else nn.Linear(node_dim, self.input_dim)
        )

        # print(f"Per_res_cov_head: node-only mode with input dim {self.input_dim}")
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.input_dim),
            nn.SiLU(),
            nn.Linear(self.input_dim, 9),
        )

    def forward(self, node_embeds: torch.Tensor):
        """
        node_embeds: (B, N, D)
        """
        assert node_embeds is not None, "node_embeds is required"

        B, N, _ = node_embeds.shape
        device = node_embeds.device

        x = self.node_proj(node_embeds)          # (B, N, H)
        raw = self.mlp(x).view(B, N, 3, 3)       # (B, N, 3, 3)

        if not self.ablate_cholesky_per_res_head:
            raw = raw.clone()
            raw = torch.tril(raw)
            diag_idx = torch.arange(3, device=device)
            diag_raw = raw[:, :, diag_idx, diag_idx]
            diag_pos = F.softplus(diag_raw) + self.eps
            raw[:, :, diag_idx, diag_idx] = diag_pos
            
        cov = raw @ raw.transpose(-1, -2)
        return cov

class pairwise_coupling_head(nn.Module):
    """
    Predict an (N, N) SPD covariance from edge features only.

    edge_embeds: (B, N, N, C)

    Output:
        cov:  (B, N, N) with cov = L @ L^T
        dccm: (B, N, N)
    """
    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int | None = None,
        eps: float = 1e-3,
        symmetrize_edges: bool = True,
        ablate_cholesky: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.symmetrize_edges = symmetrize_edges
        self.ablate_cholesky = ablate_cholesky

        self.input_dim = edge_dim if hidden_dim is None else hidden_dim
        self.edge_proj = (
            nn.Identity()
            if self.input_dim == edge_dim
            else nn.Linear(edge_dim, self.input_dim)
        )

        # print(f"pairwise_coupling_head: edge-only mode with input dim {self.input_dim}")

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.input_dim),
            nn.SiLU(),
            nn.Linear(self.input_dim, 1),
        )

    def forward(self, edge_embeds: torch.Tensor):
        """
        edge_embeds: (B, N, N, C)
        """
        assert edge_embeds is not None, "edge_embeds is required"

        B, N, N2, _ = edge_embeds.shape
        assert N == N2, "edge_embeds must have shape (B, N, N, C)"

        device = edge_embeds.device

        if self.symmetrize_edges:
            edge_embeds = 0.5 * (edge_embeds + edge_embeds.transpose(1, 2))

        x = self.edge_proj(edge_embeds)         # (B, N, N, H)
        raw = self.mlp(x).squeeze(-1)           # (B, N, N)

        if not self.ablate_cholesky:
            raw = torch.tril(raw)
            diag_idx = torch.arange(N, device=device)
            diag_raw = raw[:, diag_idx, diag_idx]
            diag_pos = F.softplus(diag_raw) + self.eps
            raw = raw.clone()
            raw[:, diag_idx, diag_idx] = diag_pos
        
        cov = raw @ raw.transpose(-1, -2)       # (B, N, N)
        dccm = compute_dccm_from_scalar_covar(cov)

        idx_dccm = torch.arange(N, device=device)
        dccm[:, idx_dccm, idx_dccm] = 1.0
        return cov, dccm