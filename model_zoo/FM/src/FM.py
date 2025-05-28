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
import torch.nn as nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, LogisticRegression, InnerProductInteraction
from fuxictr.pytorch import GEN

class FM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="FM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 regularizer=None, 
                 **kwargs):
        super(FM, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer, 
                                 net_regularizer=regularizer,
                                 **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.fm_layer = InnerProductInteraction(feature_map.num_fields, output="product_sum")
        self.lr_layer = LogisticRegression(feature_map, use_bias=True)
        
        self.symmetric = kwargs.get("symmetric", False)
        self.num_field = feature_map.num_fields
        self.gen = GEN(feature_map, embedding_dim, **kwargs)
        self.cov_reg = kwargs.get("cov_reg", 0.0000)
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        # Code for analyzing
        self.record_feature_emb = []
        self.record_y_pred = []
        self.record_feature_emb_gating = []
        self.record_gating = []
        self.record_gating_linear = []
        self.record_final_representation = []
        self.record_left = []
        self.record_right = []

    def compute_loss(self, return_dict, y_true):
        if return_dict["y_pred"].shape[-1] == 1:
            loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        else:
            loss = self.loss_fn(return_dict["y_pred"].flatten(), y_true.repeat(1, self.num_field).flatten(), reduction='mean')
        loss += self.regularization_loss()
        loss += self.cov_loss
        loss += self.infonce_loss
        return loss

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        self.grad_var_list = []
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        # self.cov_loss = self.compute_cov_loss(feature_emb.flatten(1), feature_emb.flatten(1))
        self.cov_loss = 0

        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())
        if self.training and self.analyzing:
            feature_emb.retain_grad()
        self.feature_embedding_grad = feature_emb

        gating, gating_linear = self.gen(feature_emb)
        # self.infonce_loss = self.compute_infonce_loss(feature_emb, gating)
        self.infonce_loss = 0
        # Code for analyzing
        if self.analyzing:
            self.record_gating.append(gating.detach().clone().cpu())
            self.record_gating_linear.append(gating_linear.detach().clone().cpu())
        if self.training and self.analyzing:
            gating.retain_grad()
            gating_linear.retain_grad()
        self.grad_var_list.append(gating)
        self.grad_var_list.append(gating_linear)

        row, col = torch.triu_indices(feature_emb.shape[1], feature_emb.shape[1], offset=1)
        # row, col = torch.cat([row, col]), torch.cat([col, row])
        left, right = gating[:, row], gating[:, col]
        if self.symmetric:
            left, right = gating[:, row], gating[:, col]
        else:
            left, right = gating[:, row], feature_emb[:, col]

        if self.analyzing:
            self.record_left.append(left.detach().clone().cpu())
            self.record_right.append(right.detach().clone().cpu())
        if self.training and self.analyzing:
            left.retain_grad()
            right.retain_grad()
        self.grad_var_list.append(left)
        self.grad_var_list.append(right)

        rst = left * right
        
        if self.analyzing:
            self.record_final_representation.append(rst.detach().clone().cpu())
        if self.training and self.analyzing:
            rst.retain_grad()
        self.grad_var_list.append(rst)

        bi_pooling_vec = rst.sum(-2)

        # Code for analyzing

        y_pred = self.lr_layer(X) + bi_pooling_vec.sum(dim=-1, keepdim=True)
        if self.analyzing:
            self.record_y_pred.append(y_pred.detach().clone().cpu())

        y_pred = self.output_activation(y_pred) # [B, 1]
        return_dict = {"y_pred": y_pred}
        return return_dict

