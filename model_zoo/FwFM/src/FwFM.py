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
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, InnerProductInteraction


class FwFM(BaseModel):
    """ The FwFM model
        Reference:
          - Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising, WWW'2018.
    """
    def __init__(self, 
                 feature_map, 
                 model_id="FwFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 regularizer=None, 
                 linear_type="FiLV", 
                 **kwargs):
        """ 
        linear_type: `LW`, `FeLV`, or `FiLV`
        """
        super(FwFM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs) 
        interact_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2)
        self.interaction_weight = nn.Linear(interact_dim, 1)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self._linear_type = linear_type
        if linear_type == "LW":
            self.linear_weight_layer = FeatureEmbedding(feature_map, 1, use_pretrain=False)
        elif linear_type == "FeLV":
            self.linear_weight_layer = FeatureEmbedding(feature_map, embedding_dim)
        elif linear_type == "FiLV":
            self.linear_weight_layer = nn.Linear(feature_map.num_fields * embedding_dim, 1, bias=False)
        else:
            raise NotImplementedError("linear_type={} is not supported.".format(linear_type))
        self.feature_gating = nn.Sequential(
            nn.Linear(feature_map.sum_emb_out_dim(), feature_map.sum_emb_out_dim()),
        ) if kwargs["concat_emb"] else nn.Linear(embedding_dim, embedding_dim)
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'prelu': nn.PReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
        }
        self.nonlinear = activation_dict[kwargs['emb_activation']]
        self.concat_emb = kwargs["concat_emb"]
        self.gamma = kwargs["gamma"]
        self.symmetric = kwargs["symmetric"]
        self.small_transformation = kwargs.get("small_transformation", False)
        
        self.cardinality = self.get_cardinality(feature_map)
        self.exp_group_idx = kwargs.get('exp_group_idx', None)
        
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_cardinality(self, feature_map):
        try:
            tmp = list(feature_map.features.values())
            cardinality = torch.tensor([_['vocab_size'] for _ in tmp])
            idx = torch.sort(cardinality, descending=True).indices
        except:
            idx = None
        return idx

    def init_record(self):
        self.record_feature_emb = []
        self.record_interacted_feature = []
        self.record_final_representation = []
        self.record_gating = []
        self.record_gating_linear = []

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        self.grad_var_list = []

        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())
        if self.training and self.analyzing:
            feature_emb.retain_grad()
        self.feature_embedding_grad = feature_emb

        if self.concat_emb:
            gating_linear = self.feature_gating(feature_emb.flatten(1)).squeeze(-1).reshape_as(feature_emb)
        else:
            if self.small_transformation:
                gating_linear = self.feature_gating(feature_emb)
            else:
                gating_linear = feature_emb
        gating = self.nonlinear(gating_linear) * self.gamma

        if self.exp_group_idx is not None:
            right_idx = 6 * (self.exp_group_idx + 1)
            gating[:, self.cardinality[0:right_idx]] = feature_emb[:, self.cardinality[0:right_idx]]

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
        rst = gating[:, row] * feature_emb[:, col]
        interacted_feature = rst * self.interaction_weight.weight.unsqueeze(-1)
        poly2_part = interacted_feature.sum(-1).sum(-1, keepdim=True)
        if self.analyzing:
            self.record_interacted_feature.append(interacted_feature.detach().clone().cpu())
            self.record_final_representation.append(rst.detach().clone().cpu())
        if self._linear_type == "LW":
            linear_weights = self.linear_weight_layer(X)
            linear_part = linear_weights.sum(dim=1)
        elif self._linear_type == "FeLV":
            linear_weights = self.linear_weight_layer(X)
            linear_part = (feature_emb * linear_weights).sum((1, 2)).view(-1, 1)
        elif self._linear_type == "FiLV":
            linear_part = self.linear_weight_layer(feature_emb.flatten(start_dim=1))
        y_pred = poly2_part + linear_part # bias added in poly2_part
        # y_pred = poly2_part
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
