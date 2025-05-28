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
from fuxictr.pytorch.layers import FeatureEmbedding, LogisticRegression


class FmFM(BaseModel):
    """ The FmFM model
        Reference:
        - FM2: Field-matrixed Factorization Machines for Recommender Systems, WWW'2021.
    """
    def __init__(self, 
                 feature_map, 
                 model_id="FmFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 regularizer=None, 
                 field_interaction_type="matrixed",
                 **kwargs):
        super(FmFM, self).__init__(feature_map, 
                                   model_id=model_id, 
                                   gpu=gpu, 
                                   embedding_regularizer=regularizer, 
                                   net_regularizer=regularizer,
                                   **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        num_fields = feature_map.num_fields
        interact_dim = int(num_fields * (num_fields - 1) / 2)
        self.field_interaction_type = field_interaction_type
        if self.field_interaction_type == "vectorized":
            self.interaction_weight = nn.Parameter(torch.Tensor(interact_dim, embedding_dim))
        elif self.field_interaction_type == "matrixed":
            self.interaction_weight = nn.Parameter(torch.Tensor(interact_dim, embedding_dim, embedding_dim))
        else:
            raise ValueError("field_interaction_type={} is not supported.".format(self.field_interaction_type))
        # self.feature_gating = FeatureSelection(feature_map, feature_map.sum_emb_out_dim(), embedding_dim, fs_hidden_units=[1000])
        self.feature_gating = nn.Sequential(
            nn.Linear(feature_map.sum_emb_out_dim(), feature_map.sum_emb_out_dim()),
        )
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'prelu': nn.PReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
        }
        self.nonlinear = activation_dict[kwargs['emb_activation']] if not kwargs['linear_fs'] else nn.Identity()
        self.use_fs = kwargs["use_fs"]
        self.concat_emb = kwargs["concat_emb"]
        self.gamma = kwargs["gamma"]
        self.symmetric = kwargs["symmetric"]
        nn.init.xavier_normal_(self.interaction_weight)
        self.lr_layer = LogisticRegression(feature_map)
        self.triu_index = torch.triu_indices(num_fields, num_fields, offset=1).to(self.device)

        self.cardinality = self.get_cardinality(feature_map)
        self.exp_group_idx = kwargs.get('exp_group_idx', None)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_interacted_feature = []
        self.record_feature_emb = []
        self.record_gating = []
        self.record_gating_linear = []
        self.record_final_representation = []

    def get_cardinality(self, feature_map):
        try:
            tmp = list(feature_map.features.values())
            cardinality = torch.tensor([_['vocab_size'] for _ in tmp])
            idx = torch.sort(cardinality, descending=True).indices
        except:
            idx = None
        return idx

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        self.grad_var_list = []
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())
        if self.training and self.analyzing:
            feature_emb.retain_grad()
        self.feature_embedding_grad = feature_emb

        if self.concat_emb:
            gating_linear = self.feature_gating(feature_emb.flatten(1)).squeeze(-1).reshape_as(feature_emb)
        else:
            gating_linear = feature_emb
        gating = self.nonlinear(gating_linear) * self.gamma
        
        if self.exp_group_idx is not None:
            right_idx = 6 * (self.exp_group_idx + 1)
            gating[:, self.cardinality[0:right_idx]] = feature_emb[:, self.cardinality[0:right_idx]]
        # gating[:, self.cardinality[:2]] = feature_emb[:, self.cardinality[0:2]]

        if self.analyzing:
            self.record_gating.append(gating.detach().clone().cpu())
            self.record_gating_linear.append(gating_linear.detach().clone().cpu())
        if self.training and self.analyzing:
            gating.retain_grad()
            gating_linear.retain_grad()
        self.grad_var_list.append(gating)
        self.grad_var_list.append(gating_linear)


        left_emb = torch.index_select(feature_emb, 1, self.triu_index[0])
        right_emb = torch.index_select(gating, 1, self.triu_index[1])
        if self.field_interaction_type == "vectorized":
            left_emb = left_emb * self.interaction_weight
        elif self.field_interaction_type == "matrixed":
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
        rst = (left_emb * right_emb)
        bi_interaction = rst.sum(-2)
        if self.analyzing:
            self.record_interacted_feature.append(rst.detach().clone().cpu())
            self.record_final_representation.append(bi_interaction.detach().clone().cpu())
        y_pred = bi_interaction.sum(dim=-1, keepdim=True)
        y_pred += self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict
