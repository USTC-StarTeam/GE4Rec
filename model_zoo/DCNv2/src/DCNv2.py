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
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch import GEN

class DCNv2(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNv2", 
                 gpu=-1,
                 model_structure="parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None, 
                 **kwargs):
        super(DCNv2, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        self.crossnet = CrossNetV2(input_dim, num_cross_layers, embedding_dim, feature_map, kwargs)
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"], \
               "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(input_dim=input_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=stacked_dnn_hidden_units,
                                         hidden_activations=dnn_activations,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            final_dim = stacked_dnn_hidden_units[-1]
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure == "crossnet_only": # only CrossNet
            final_dim = input_dim
            
        # self.fc = nn.Linear(final_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(final_dim, final_dim * 2),
            nn.ReLU(),
            nn.Linear(final_dim * 2, 1),
        )
        if kwargs.get('use_high_cardinality_predict', False) or kwargs.get('use_low_cardinality_predict', False) or kwargs.get('use_random_predict', False):
            num_target = kwargs.get('num_target', 1)
            self.fc = nn.Linear(embedding_dim * num_target + parallel_dnn_hidden_units[-1], 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_feature_emb = []
        self.record_final_out = []
        self.record_feature_emb = []
        self.record_gating = []
        self.record_gating_linear = []
        self.record_final_representation = []

    def compute_loss(self, return_dict, y_true):
        loss = super().compute_loss(return_dict, y_true)
        return loss

    def forward(self, inputs):
        self.grad_var_list = []
        X = self.get_inputs(inputs)

        feature_emb = self.embedding_layer(X, flatten_emb=True)
        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())

        cross_out = self.crossnet(feature_emb)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(feature_emb)], dim=-1)
        if self.analyzing:
            self.record_final_out.append(final_out.detach().clone().cpu())
        y_pred = self.fc(final_out)

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers, embedding_dim=None, feature_map=None, kwargs=None):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim, bias=False)
                                          for _ in range(self.num_layers))
        self.transform_layers = nn.ModuleList(nn.Linear(input_dim, embedding_dim, bias=False)
                                              for _ in range(self.num_layers - 1))
        self.batch_norm_layers = nn.ModuleList(nn.BatchNorm1d(1)
                                          for _ in range(self.num_layers))
        self.embedding_dim = embedding_dim
        self.num_field = input_dim // embedding_dim
        self.kwargs = kwargs
        self.symmetric = kwargs.get("symmetric", False)
        self.cov_diag = kwargs.get("cov_diag", None)
        self.cov_nondiag = kwargs.get("cov_nondiag", None)

        self.gen_layers = nn.ModuleList(GEN(feature_map, embedding_dim, **kwargs) for _ in range(self.num_layers))
        self.bn = nn.BatchNorm1d(self.num_field * self.embedding_dim, affine=False)
        self.cardinality = self.get_cardinality(feature_map)

    def get_cardinality(self, feature_map):
        tmp = list(feature_map.features.values())
        cardinality = torch.tensor([_.get('vocab_size', 0) for _ in tmp])
        idx = torch.sort(cardinality, descending=True).indices
        return idx

    def init_record(self):
        self.record_cross_emb = []
        self.record_cross_emb_residual = []
        self.record_non_linear_rep = []
        self.record_cross_rst = []
        self.record_X_0 = []
        self.record_left = []
        self.record_right = []

    def forward(self, feature_embedding, gating=None):
        self.cov_loss = 0
        self.infonce_loss = 0
        X_0 = feature_embedding
        X_i = feature_embedding # b x dim

        for i in range(self.num_layers):
            if self.kwargs.get('use_gen_rst', False):
                X_0 = self.gen_layers[i](X_i)[0]
            else:
                X_0 = self.gen_layers[i](feature_embedding)[0]
            non_linear_rep = X_i
            tmp = self.cross_layers[i](non_linear_rep)

            if self.analyzing:
                self.record_X_0.append(X_0.detach().clone().cpu())
                self.record_non_linear_rep.append(non_linear_rep.detach().clone().cpu())
                self.record_left.append(X_0.detach().clone().cpu())
                self.record_right.append(tmp.detach().clone().cpu())
            tmp = X_0 * tmp
            X_i = X_i + tmp
            X_i = torch.nn.functional.dropout(
                X_i,
                p=self.kwargs.get('cross_drop', 0),
                training=self.training,
            )
            if self.analyzing:
                self.record_cross_emb.append(tmp.detach().clone().cpu())
                self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
        return X_i


    def forward(self, feature_embedding, gating=None):
        self.cov_loss = 0
        self.infonce_loss = 0
        X_0 = feature_embedding
        X_i = feature_embedding # b x dim
        rst = 0

        for i in range(self.num_layers):
            if self.kwargs.get('use_gen_rst', False):
                X_0 = self.gen_layers[i](X_i)[0]
            else:
                X_0 = self.gen_layers[i](feature_embedding)[0]
            non_linear_rep = X_i
            tmp = self.cross_layers[i](non_linear_rep)

            if self.analyzing:
                self.record_X_0.append(X_0.detach().clone().cpu())
                self.record_non_linear_rep.append(non_linear_rep.detach().clone().cpu())
                self.record_left.append(X_0.detach().clone().cpu())
                self.record_right.append(tmp.detach().clone().cpu())
            tmp = X_0 * tmp
            if self.kwargs.get('use_random_predict', False):
                num_target = self.kwargs.get('num_target', 5)
                if self.training:
                    selected_idx = torch.randperm(self.num_field)[torch.arange(num_target)]
                else:
                    selected_idx = self.cardinality[:num_target]
                rst += tmp.reshape(-1, self.num_field, self.embedding_dim)[:, selected_idx].flatten(1)
            if self.kwargs.get('use_high_cardinality_predict', False):
                num_target = self.kwargs.get('num_target', 5)
                selected_idx = self.cardinality[:num_target]
                rst += tmp.reshape(-1, self.num_field, self.embedding_dim)[:, selected_idx].flatten(1)
            if self.kwargs.get('use_low_cardinality_predict', False):
                num_target = self.kwargs.get('num_target', 5)
                selected_idx = self.cardinality[-num_target:]
                rst += tmp.reshape(-1, self.num_field, self.embedding_dim)[:, selected_idx].flatten(1)
            if self.kwargs.get('use_hard_mask_predict', False):
                num_target = self.kwargs.get('num_target', 1)
                if self.training:
                    selected_idx = torch.randperm(self.num_field)[torch.arange(num_target)]
                else:
                    selected_idx = self.cardinality[:num_target]
                rst += tmp.reshape(-1, self.num_field, self.embedding_dim)[:, selected_idx].flatten(1)
            X_i = X_i + tmp
            X_i = torch.nn.functional.dropout(
                X_i,
                p=self.kwargs.get('cross_drop', 0),
                training=self.training,
            )
            if self.analyzing:
                self.record_cross_emb.append(tmp.detach().clone().cpu())
                self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
        if self.kwargs.get('use_high_cardinality_predict', False) or self.kwargs.get('use_low_cardinality_predict', False) or self.kwargs.get('use_random_predict', False)  or self.kwargs.get('use_low_cardinality_predict', False) or self.kwargs.get('use_hard_mask_predict', False):
            return rst
        return X_i
