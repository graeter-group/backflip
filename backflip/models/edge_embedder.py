import torch
from torch import nn
from backflip.models.utils import get_index_embedding, calc_distogram

class EdgeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.ablate_edges = getattr(self._cfg, "ablate_edges", False)

        if "embed_breaks" in self._cfg:
            self.embed_breaks = self._cfg.embed_breaks
        else:
            self.embed_breaks = False

        if self.embed_breaks:
            self.c_breaks = 2
        else:
            self.c_breaks = 0

        if "embed_distance_encoding" in self._cfg:
            self.embed_distance_encoding = self._cfg.embed_distance_encoding
        else:
            self.embed_distance_encoding = True  # Default to True for backward compatibility

        if "embed_esmfold_pair_repr" in self._cfg:
            self.embed_esmfold_pair_repr = self._cfg.embed_esmfold_pair_repr
        else:
            self.embed_esmfold_pair_repr = False

        # Validate: at least one embedder must be enabled unless we explicitly ablate
        if not self.embed_distance_encoding and not self.embed_esmfold_pair_repr:
            self.ablate_edges = True
            print(f'Ablating edges since no edge features are enabled!')
		
        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        # Only load dimension parameters if the corresponding features are enabled
        if self.embed_distance_encoding:
            if "distance_encoding_dim" in self._cfg:
                self.distance_encoding_dim = self._cfg.distance_encoding_dim
            else:
                self.distance_encoding_dim = 64  # Default value for old checkpoints

        if self.embed_esmfold_pair_repr:
            if "esmfold_pair_repr_dim" in self._cfg:
                self.esmfold_pair_repr_dim = self._cfg.esmfold_pair_repr_dim
            else:
                self.esmfold_pair_repr_dim = 128

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        # Create embedders only if they are enabled
        if self.embed_distance_encoding:
            distance_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
            self.distance_embedder = nn.Sequential(
                nn.Linear(distance_edge_feats, self.distance_encoding_dim),
                nn.ReLU(),
                nn.Linear(self.distance_encoding_dim, self.distance_encoding_dim),
                nn.ReLU(),
                nn.Linear(self.distance_encoding_dim, self.distance_encoding_dim),
                nn.LayerNorm(self.distance_encoding_dim),
            )

        if self.embed_esmfold_pair_repr:
            esmfold_pair_repr_edge_feats = self.feat_dim * 3 + self._cfg.esmfold_pair_repr_dim
            self.esmfold_pair_repr_embedder = nn.Sequential(
                nn.Linear(esmfold_pair_repr_edge_feats, self.esmfold_pair_repr_dim),
                nn.ReLU(),
                nn.Linear(self.esmfold_pair_repr_dim, self.esmfold_pair_repr_dim),
                nn.ReLU(),
                nn.Linear(self.esmfold_pair_repr_dim, self.esmfold_pair_repr_dim),
                nn.LayerNorm(self.esmfold_pair_repr_dim),
            )

        total_edge_feats = 0
        if self.embed_distance_encoding:
            total_edge_feats += self.distance_encoding_dim
        if self.embed_esmfold_pair_repr:
            total_edge_feats += self.esmfold_pair_repr_dim

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, pos, breaks=None):
        rel_pos = pos[:, :, None] - pos[:, None, :]
        pos_emb = get_index_embedding(rel_pos, self._cfg.feat_dim - self.c_breaks, max_len=2056)

        # Adding breaks here enables to keep the same feat_dim and thus the same parameter set as without breaks thus making both models backwards compatible
        if self.embed_breaks:
            if breaks is None:
                breaks = torch.zeros([pos.shape[0], pos.shape[1], pos.shape[1]], device=pos.device, dtype=torch.float32)
            else:
                breaks = breaks[:, :, None] * breaks[:, None, :] * (torch.diag(torch.ones(pos.shape[1] - 1, device=pos.device, dtype=torch.float32), diagonal=1) + torch.diag(torch.ones(pos.shape[1] - 1, device=pos.device, dtype=torch.float32), diagonal=-1)).unsqueeze(0)

            pos_emb = torch.cat([pos_emb, breaks.unsqueeze(-1), (1 - breaks).unsqueeze(-1)], dim=-1)

        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, p_mask, res_idx=None, breaks=None, esmfold_pair_repr=None):
        num_batch, num_res, _ = s.shape

        if self.ablate_edges:
            edge_feats = torch.zeros(
                num_batch, num_res, num_res, self.c_p,
                device=s.device,
                dtype=s.dtype,
            )
            edge_feats *= p_mask.unsqueeze(-1)
            return edge_feats

        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)
        if res_idx is None:
            pos = torch.arange(
                num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        else:
            pos = res_idx.float()

        relpos_feats = self.embed_relpos(pos, breaks=breaks)

        if self.embed_distance_encoding:
            # use position in space encoding
            dist_feats = calc_distogram(
                t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
            sc_feats = calc_distogram(
                sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
            distance_encoding_edge_feats = torch.concat([cross_node_feats, relpos_feats, dist_feats, sc_feats], dim=-1)
        if self.embed_esmfold_pair_repr:	
            # use esmfold pair repr encoding instead of position in space encoding if embed_esmfold_pair_repr is True
            if esmfold_pair_repr is not None:
                esmfold_pair_repr_feats = esmfold_pair_repr
            else:
                esmfold_pair_repr_feats = torch.zeros_like(t)
            esmfold_pair_repr_edge_feats = torch.concat([cross_node_feats, relpos_feats, esmfold_pair_repr_feats], dim=-1)

        # Build list of edge features to concatenate
        edge_feat_list = []
        if self.embed_distance_encoding:
            distance_encoding_edge_feats = self.distance_embedder(distance_encoding_edge_feats)
            edge_feat_list.append(distance_encoding_edge_feats)
        if self.embed_esmfold_pair_repr:
            esmfold_pair_repr_edge_feats = self.esmfold_pair_repr_embedder(esmfold_pair_repr_edge_feats)
            edge_feat_list.append(esmfold_pair_repr_edge_feats)

        total_edge_feats = torch.cat(edge_feat_list, dim=-1)
        edge_feats = self.edge_embedder(total_edge_feats)
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats