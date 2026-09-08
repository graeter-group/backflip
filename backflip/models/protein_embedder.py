# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2024 HITS gGmbH.
# Licensed under the MIT license.

"""Neural network architecture for the flow model."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backflip.models import ipa_pytorch
from backflip.models.node_embedder import NodeEmbedder
from backflip.models.edge_embedder import EdgeEmbedder
from backflip.data import utils as du

class ProteinEmbedderIPA(nn.Module):

    def __init__(self, model_conf):
        super().__init__()

        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_embedder = NodeEmbedder(model_conf.node_features)
        self.edge_embedder = EdgeEmbedder(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )

            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            
            # NOTE: no bb update here
            # self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
            #     self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )
    def forward(self, input_feats):
    
        if 'res_idx' not in input_feats:
            res_idx = None
        else:
            res_idx = input_feats['res_idx']

        if 'breaks' in input_feats:
            breaks = input_feats['breaks']
        else:
            breaks = None

        if 'aatype' in input_feats:
            # assumes this contains integers from 0-19
            aatype = input_feats['aatype']
        else:
            aatype = None

        if 'esm_emb' in input_feats:
            esm_embedding = input_feats['esm_emb']
        else:
            esm_embedding = None

        if 'esmfold_s_z' in input_feats:
            esmfold_pair_repr = input_feats['esmfold_s_z']
        else:
            esmfold_pair_repr = None

        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        trans = input_feats['trans_1']
        rotmats = input_feats['rotmats_1']

        # Initialize node and edge embeddings
        node_res_idxs = input_feats['res_idx'] if self._model_conf.node_features.embed_res_idx else torch.zeros_like(node_mask)
        edge_res_idxs = input_feats['res_idx'] if self._model_conf.edge_features.embed_res_idx else torch.zeros_like(node_mask)
        init_node_embed = self.node_embedder(node_mask, res_idx=node_res_idxs, breaks=breaks, aatype=aatype, esm_embedding=esm_embedding)

        trans_sc = torch.zeros_like(trans)
        init_edge_embed = self.edge_embedder(
            init_node_embed, trans, trans_sc, edge_mask, res_idx=edge_res_idxs, breaks=breaks, esmfold_pair_repr=esmfold_pair_repr)
        
        # Initial rigids
        curr_rigids = du.create_rigid(rotmats, trans,)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            
            # NOTE: no bb update here
            # rigid_update = self.trunk[f'bb_update_{b}'](
            #     node_embed * node_mask[..., None])
            # curr_rigids = curr_rigids.compose_q_update_vec(
            #     rigid_update, (node_mask * node_mask)[..., None])
            
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        return node_embed, edge_embed