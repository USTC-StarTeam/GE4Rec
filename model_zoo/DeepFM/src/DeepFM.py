# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, FactorizationMachine, InnerProductInteraction, LogisticRegression
from fuxictr.pytorch import GEN
from fuxictr.pytorch.torch_utils import get_optimizer, get_loss

class DeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 use_deepfm1=False,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_field = feature_map.num_fields
        self.fm_layer = InnerProductInteraction(feature_map.num_fields, output="product_sum")
        self.lr_layer = LogisticRegression(feature_map, use_bias=True)
        self.mlp = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=None, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm)
        self.linear_out = nn.Linear(hidden_units[-1], 1)
        self.gen = GEN(feature_map, embedding_dim, **kwargs)
        self.symmetric = kwargs.get("symmetric", False)
        self.cov_diag = kwargs.get("cov_diag", None)
        self.cov_nondiag = kwargs.get("cov_nondiag", None)
        self.cardinality = self.get_cardinality(feature_map)
        self.kwargs = kwargs
        if self.kwargs.get('use_masked_predict', False):
            self.mask_emb = nn.Embedding(1, embedding_dim)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_feature_emb = []
        self.record_bi_pooling_vec = []
        self.record_gating = []
        self.record_gating_linear = []
        self.record_final_representation = []
        self.record_dnn_out = []

    def compute_infonce_loss(self, A, B):
        batch_size, num_field = A.shape[0], A.shape[1]
        A, B = torch.nn.functional.normalize(A, p=2, dim=-1), torch.nn.functional.normalize(B, p=2, dim=-1)
        sim_ij = torch.bmm(A, B.permute(0, 2, 1)) / 1
        sim_ji = torch.bmm(B, A.permute(0, 2, 1)) / 1
        labels = torch.arange(num_field, dtype=torch.long).to(A.device).unsqueeze(0).repeat(batch_size, 1)
        loss_i = torch.nn.functional.cross_entropy(sim_ij, labels)
        loss_j = torch.nn.functional.cross_entropy(sim_ji, labels)
        return (loss_i + loss_j) / 2.0 * 0.001
    
    def get_cardinality(self, feature_map):
        tmp = list(feature_map.features.values())
        cardinality = torch.tensor([_.get('vocab_size', 0) for _ in tmp])
        idx = torch.sort(cardinality, descending=True).indices
        return idx

    # def forward(self, inputs):
    #     """
    #     Inputs: [X,y]
    #     """
    #     self.grad_var_list = []
    #     X = self.get_inputs(inputs)
    #     feature_emb = self.embedding_layer(X)

    #     if self.training and self.analyzing:
    #         feature_emb.retain_grad()
    #     self.feature_embedding_grad = feature_emb
    #     if self.analyzing:
    #         self.record_feature_emb.append(feature_emb.detach().clone().cpu())

    #     gating, _ = self.gen(feature_emb, X)
    #     if self.analyzing:
    #         self.record_gating.append(gating.detach().clone().cpu())
        
    #     # self.cov_loss = self.compute_cov_loss(feature_emb.flatten(1), gating.flatten(1))
    #     self.cov_loss = 0
    #     # self.infonce_loss = self.compute_infonce_loss(feature_emb, gating)
    #     self.infonce_loss = 0

    #     row, col = torch.triu_indices(feature_emb.shape[1], feature_emb.shape[1], offset=1)
    #     # row, col = torch.arange(self.num_field).repeat_interleave(self.num_field), torch.arange(self.num_field).repeat(self.num_field)
    #     if not self.symmetric:
    #         rst = gating[:, row] * feature_emb[:, col]
    #     else:
    #         rst = gating[:, row] * gating[:, col]
    #     bi_pooling_vec = rst.sum(-2)
    #     if self.analyzing:
    #         self.record_bi_pooling_vec.append(bi_pooling_vec.detach().clone().cpu())
    #         self.record_final_representation.append(bi_pooling_vec.detach().clone().cpu())
    #     if self.training and self.analyzing:
    #         bi_pooling_vec.retain_grad()
    #     self.grad_var_list.append(bi_pooling_vec)

    #     y_pred = self.lr_layer(X) + bi_pooling_vec.sum(dim=-1, keepdim=True)

    #     dnn_out = self.mlp(feature_emb.flatten(start_dim=1))
    #     dnn_out = dnn_out * self.linear_out.weight.T.unsqueeze(0).squeeze(-1) + self.linear_out.bias / dnn_out.shape[-1]
    #     if self.analyzing:
    #         self.record_dnn_out.append(dnn_out.detach().clone().cpu())
    #     dnn_out = dnn_out.sum(-1, keepdim=True)
    #     y_pred += dnn_out
    #     self.grad_var_list.extend(self.mlp.grad_var_list)
        
    #     y_pred = self.output_activation(y_pred)
    #     return_dict = {"y_pred": y_pred}
    #     return return_dict

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        self.grad_var_list = []
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X) # [B, num_field, embed_dim]

        gating, _ = self.gen(feature_emb, X)
        if self.kwargs.get('use_masked_predict', False) and self.training:
            num_target = self.kwargs.get('num_target', 5)
            if self.training:
                selected_idx = torch.randperm(self.num_field)[torch.arange(num_target)]
            else:
                selected_idx = self.cardinality[:num_target]
            mask = torch.zeros_like(feature_emb, device=feature_emb.device)
            mask[:, selected_idx] = self.mask_emb.weight
            feature_emb = feature_emb + mask
        
        self.cov_loss = 0
        self.infonce_loss = 0

        row, col = torch.arange(self.num_field).repeat_interleave(self.num_field), torch.arange(self.num_field).repeat(self.num_field)
        # rst = gating[:, row] * feature_emb[:, col]
        rst = feature_emb[:, row] * gating[:, col]
        rst = rst.reshape(-1, self.num_field, self.num_field, rst.shape[-1])
        # rst = rst.sum(1)
        if self.kwargs.get('use_random_predict', False):
            if self.training:
                selected_idx = torch.arange(15)
                random_field = torch.randperm(self.num_field)[selected_idx]
                rst = rst[:, random_field]
                rst = rst.sum(1)
            else:
                rst = rst.sum(1) / self.num_field * 15
        elif self.kwargs.get('use_high_cardinality_predict', False):
            num_target = self.kwargs.get('num_target', 5)
            selected_idx = self.cardinality[:num_target]
            rst = rst[:, selected_idx]
            rst = rst.sum(1)
        elif self.kwargs.get('use_low_cardinality_predict', False):
            num_target = self.kwargs.get('num_target', 5)
            selected_idx = self.cardinality[-num_target:]
            rst = rst[:, selected_idx]
            rst = rst.sum(1)
        elif self.kwargs.get('use_masked_predict', False):
            rst = rst.sum(1)
        else:
            rst = rst.sum(1)
        bi_pooling_vec = rst.sum(-2)

        y_pred = self.lr_layer(X) + bi_pooling_vec.sum(dim=-1, keepdim=True)

        dnn_out = self.mlp(feature_emb.flatten(start_dim=1))
        dnn_out = dnn_out * self.linear_out.weight.T.unsqueeze(0).squeeze(-1) + self.linear_out.bias / dnn_out.shape[-1]
        dnn_out = dnn_out.sum(-1, keepdim=True)
        y_pred += dnn_out
        
        y_pred = self.output_activation(y_pred)
        # if not self.training:
        #     y_pred = y_pred[..., -1:]
            # y_pred = torch.mean(y_pred, dim=-1, keepdim=True)
        return_dict = {"y_pred": y_pred}
        return return_dict
