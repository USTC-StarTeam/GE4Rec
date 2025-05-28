# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (C) 2021 The DeepCTR-Torch authors for CrossNetMix
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
from ..attentions import ScaledDotProductAttention
from fuxictr.pytorch.layers import FeatureEmbedding


class CrossInteraction(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteraction, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interact_out = self.weight(X_i) * X_0 + self.bias
        return interact_out


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteraction(input_dim)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


# class CrossNetV2(nn.Module):
#     def __init__(self, input_dim, num_layers):
#         super(CrossNetV2, self).__init__()
#         self.num_layers = num_layers
#         self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
#                                           for _ in range(self.num_layers))
#         # self.cross_layers_nonlinear = nn.ModuleList(nn.Linear(input_dim, input_dim)
#         #                                   for _ in range(self.num_layers))
#         self.batch_norm_layers = nn.ModuleList(nn.BatchNorm1d(1)
#                                           for _ in range(self.num_layers))

#     def init_record(self):
#         self.record_cross_emb = []
#         self.record_cross_emb_residual = []

#     def forward(self, feature_embedding, gating=None):
#         X_0 = feature_embedding
#         if gating is not None:
#             X_0 = gating
#         X_i = feature_embedding # b x dim
#         if self.analyzing:
#             self.record_cross_emb.append(X_i.detach().clone().cpu())
#             self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
#         for i in range(self.num_layers):
#             # tmp = X_i
#             # tmp = self.cross_layers[i](tmp)
#             # # tmp = self.batch_norm_layers[i](tmp.unsqueeze(1)).squeeze()
#             # # tmp = torch.nn.functional.tanh(tmp)
#             # tmp = X_0 * tmp
#             # X_i = X_i + tmp

#             tmp = X_0
#             # tmp = self.cross_layers_nonlinear[i](tmp)
#             # tmp = torch.tanh(tmp)
#             tmp = self.cross_layers[i](tmp)
#             # tmp = torch.nn.functional.relu(tmp) * 0.5
#             # tmp = torch.nn.functional.tanh(tmp) * 6 # best for DCNv2-avazu
#             tmp = X_i * tmp
#             # tmp = self.batch_norm_layers[i](X_i.unsqueeze(1)).squeeze() * tmp
#             X_i = X_i + tmp

#             if self.analyzing:
#                 self.record_cross_emb.append(tmp.detach().clone().cpu())
#                 self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
#         return X_i

# class CrossNetV2(nn.Module):
#     def __init__(self, input_dim, num_layers, embedding_dim=None):
#         super(CrossNetV2, self).__init__()
#         self.num_layers = num_layers
#         self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
#                                           for _ in range(self.num_layers))
#         self.batch_norm_layers = nn.ModuleList(nn.BatchNorm1d(1)
#                                           for _ in range(self.num_layers))

#     def init_record(self):
#         self.record_cross_emb = []
#         self.record_cross_emb_residual = []

#     def forward(self, feature_embedding, gating=None):
#         X_0 = feature_embedding
#         if gating is not None:
#             X_0 = gating
#         X_i = feature_embedding # b x dim
#         if self.analyzing:
#             self.record_cross_emb.append(X_i.detach().clone().cpu())
#             self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
#         for i in range(self.num_layers):
#             tmp = X_i
#             tmp = self.cross_layers[i](tmp)
#             # tmp = self.batch_norm_layers[i](tmp.unsqueeze(1)).squeeze()
#             tmp = torch.nn.functional.relu(tmp) * 0.1 # best for DCNv2 on criteo
#             # tmp = torch.nn.functional.relu(tmp) * 0.5 # best for DCNv2 on criteo
#             # tmp = torch.nn.functional.tanh(tmp) * 8 # best for DCNv2 on avazu
#             tmp = X_0 * tmp
#             X_i = X_i + tmp
#             if self.analyzing:
#                 self.record_cross_emb.append(tmp.detach().clone().cpu())
#                 self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
#         return X_i

# class CrossNetV2(nn.Module):
#     def __init__(self, input_dim, num_layers, embedding_dim=None):
#         super(CrossNetV2, self).__init__()
#         self.num_layers = num_layers
#         self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
#                                           for _ in range(self.num_layers))
#         self.transform_layers = nn.ModuleList(nn.Linear(input_dim, embedding_dim, bias=False)
#                                               for _ in range(self.num_layers - 1))
#         self.batch_norm_layers = nn.ModuleList(nn.BatchNorm1d(1)
#                                           for _ in range(self.num_layers))
#         self.embedding_dim = embedding_dim
#         self.num_field = input_dim // embedding_dim
#         self.transform_input = nn.Linear(input_dim, input_dim, bias=False)

#     def init_record(self):
#         self.record_cross_emb = []
#         self.record_cross_emb_residual = []

#     def forward(self, feature_embedding, gating=None):
#         X_0 = feature_embedding
#         if gating is not None:
#             X_0 = gating
#         X_i = feature_embedding # b x dim
#         if self.analyzing:
#             self.record_cross_emb.append(X_i.detach().clone().cpu())
#             self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
#         for i in range(self.num_layers):
#             if i == 0:
#                 tmp = X_i
#                 tmp = self.transform_input(tmp)
#             else:
#                 tmp = X_i.reshape(-1, self.num_field, self.embedding_dim)
#                 tmp = torch.einsum('bfd,fde->bfe', tmp, self.transform_layers[i - 1].weight.reshape(self.num_field, self.embedding_dim, self.embedding_dim)).flatten(1)
#             tmp = torch.nn.functional.relu(tmp) * 0.5
#             tmp = self.cross_layers[i](tmp)
#             tmp = X_0 * tmp
#             X_i = X_i + tmp
#             if self.analyzing:
#                 self.record_cross_emb.append(tmp.detach().clone().cpu())
#                 self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
#         return X_i

class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i

# class CrossNetV2(nn.Module):
#     def __init__(self, input_dim, num_layers):
#         super(CrossNetV2, self).__init__()
#         self.num_layers = num_layers
#         self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
#                                           for _ in range(self.num_layers))

#     def forward(self, X_0):
#         X_i = X_0 # b x dim
#         for i in range(self.num_layers):
#             X_i = X_i + X_0 * self.cross_layers[i](X_i)
#         return X_i

class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X):
        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        return output, attention

class FeatureSelection(nn.Module):
    def __init__(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[], 
                 fs1_context=['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']):
        fs1_context=['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
        # fs1_context = ['C' + str(idx) for idx in range(1, 27)]
        # fs1_context += ['C' + str(idx) for idx in range(1, 27)]
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                            required_feature_columns=fs1_context)
        # self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim)
        self.fs1_gate = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.cnt = 0

    def forward(self, X, flat_emb):
        self.cnt += 1
        fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        gt1 = self.fs1_gate(fs1_input) * 2
        if self.cnt % 4000 == 0:
            print(gt1)
            torch.save(gt1, f'gate_real_all_{self.cnt}.pt')
        feature1 = flat_emb * gt1
        return feature1

class CrossNetV2_FS(nn.Module):
    def __init__(self, input_dim, num_layers, feature_map, embedding_dim):
        super(CrossNetV2_FS, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))
        # self.feature_gating = FeatureSelection(feature_map, input_dim, embedding_dim, fs_hidden_units=[128, 128])

        fs1_context = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
        self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                            required_feature_columns=fs1_context)
        # self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim)
        self.fs_gate = nn.ModuleList([
            nn.Sequential(
            nn.Linear(embedding_dim * len(fs1_context), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        ) for _ in range(num_layers)
        ])

    def init_record(self):
        self.record_cross_emb = []
        self.record_cross_emb_residual = []

    def forward(self, X_0):
        # X_0 = self.feature_gating(self.input_data, X_0)
        X_i = X_0 # b x dim
        # gating = self.fs_gate[0](self.fs1_ctx_emb(self.input_data).flatten(1)) * 2
        # X_0, X_i = X_0 * gating, X_i * gating
        if self.analyzing:
            self.record_cross_emb.append(X_i.detach().clone().cpu())
            self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
        for i in range(self.num_layers):
            gating = self.fs_gate[i](self.fs1_ctx_emb(self.input_data).flatten(1)) * 2
            X_0 = X_0 * gating
            X_i = X_i * gating
            tmp = X_0 * self.cross_layers[i](X_i)
            X_i = X_i + tmp
            if self.analyzing:
                self.record_cross_emb.append(tmp.detach().clone().cpu())
                self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
        return X_i

class CrossNetV2_att(nn.Module):
    def __init__(self, input_dim, num_layers, aux_embedding_dim, feature_map=None):
        super(CrossNetV2_att, self).__init__()
        self.num_layers = num_layers
        self.aux_embedding_dim = aux_embedding_dim
        self.num_field = input_dim // aux_embedding_dim
        interact_dim = self.num_field * self.num_field
        self.interaction_weight = nn.Parameter(torch.Tensor(interact_dim, aux_embedding_dim, aux_embedding_dim))
        nn.init.xavier_normal_(self.interaction_weight)
        # self.att_embedding_layer = FeatureEmbedding(feature_map, aux_embedding_dim)
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))
        # self.att_layers = nn.ModuleList(
        #     MultiHeadSelfAttention(self.aux_embedding_dim,
        #     attention_dim=self.aux_embedding_dim,
        #     num_heads=1,
        #     dropout_rate=0.1,
        #     use_residual=True,
        #     use_scale=True,
        #     layer_norm=True
        #     ) for _ in range(self.num_layers)
        # )
        self.W_q = nn.Parameter(torch.Tensor(self.num_field, aux_embedding_dim, aux_embedding_dim))
        self.W_k = nn.Parameter(torch.Tensor(self.num_field, aux_embedding_dim, aux_embedding_dim))
        nn.init.xavier_normal_(self.W_q)
        nn.init.xavier_normal_(self.W_k)
        # self.W_q = nn.Linear(aux_embedding_dim, aux_embedding_dim, bias=False)
        # self.W_k = nn.Linear(aux_embedding_dim, aux_embedding_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(0.1)
        self.cnt = 0
        self.att_layer_norm = nn.LayerNorm(aux_embedding_dim)
        # self.att_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs1_context)),
        #                           output_dim=1,
        #                           hidden_units=fs_hidden_units,
        #                           hidden_activations="ReLU",
        #                           output_activation="Sigmoid",
        #                           batch_norm=False)
        self.att_gate = nn.Sequential(
            nn.Linear(aux_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def init_record(self):
        self.record_cross_emb = []
        self.record_cross_emb_residual = []

    # def forward(self, X_0):
    #     X_i = X_0 # b x dim
    #     if self.analyzing:
    #         self.record_cross_emb.append(X_i.detach().clone().cpu())
    #         self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
    #     # X_i = X_i.reshape(X_i.shape[0], -1, self.aux_embedding_dim)
    #     # B, F, D = X_i.shape
    #     # right_emb = X_i.repeat(1, F, 1)
    #     # left_emb = X_0.repeat_interleave(F, dim=1)
    #     # left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
    #     for i in range(self.num_layers):
    #         attn = self.att_layers[i](X_i.reshape(X_0.shape[0], -1, self.aux_embedding_dim))[1].mean(1)
    #         # attn = attn.repeat_interleave(self.aux_embedding_dim, dim=1).repeat_interleave(self.aux_embedding_dim, dim=2)
    #         tmp = X_0 * (X_i @ (self.cross_layers[i].weight) + self.cross_layers[i].bias)
    #         # tmp = X_0 * (X_i @ (self.cross_layers[i].weight * attn) + self.cross_layers[i].bias)
    #         # tmp = X_0 * (torch.einsum('bd,bde->be', X_i, self.cross_layers[i].weight * attn) + self.cross_layers[i].bias)
    #         # tmp = X_0 * (torch.einsum('bd,bde->be', X_i, attn) + self.cross_layers[i].bias)
    #         # tmp = X_0 * (torch.einsum('bd,de->be', X_i, nn.functional.normalize(self.cross_layers[i].weight, dim=0)) + self.cross_layers[i].bias)
    #         # tmp = X_0 * self.cross_layers[i](X_i)
            
    #         X_i = X_i + tmp
    #         if self.analyzing:
    #             self.record_cross_emb.append(tmp.detach().clone().cpu())
    #             self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
    #     return X_i
    
    def forward(self, X_0):
        def scale_dot_attention(Q, K, scale=1):
            scores = torch.matmul(Q, K.transpose(-1, -2))
            scores = scores / scale
            attention = scores.softmax(dim=-1)
            # attention = torch.nn.functional.normalize(scores, dim=-1)
            # attention = torch.nn.functional.silu(scores)
            # attention = torch.sigmoid(scores) * 2
            attention = nn.functional.dropout(attention, p=0.0, training=self.training)
            return attention
        X_0 = X_0.reshape(X_0.shape[0], -1, self.aux_embedding_dim)
        X_i = X_0 # b x dim
        if self.analyzing:
            self.record_cross_emb.append(X_i.detach().clone().cpu())
            self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
        B, F, D = X_i.shape
        left_emb = X_i.repeat_interleave(F, dim=1)
        self.cnt += 1
        for i in range(self.num_layers):
            att_emb = self.att_layer_norm(self.att_emb)
            q, k = torch.einsum('bfd, fde->bfe', att_emb, self.W_q), torch.einsum('bfd, fde->bfe', att_emb, self.W_k)
            # q, k = self.W_q(att_emb), self.W_k(att_emb)
            # q, k = att_emb, att_emb
            attn = scale_dot_attention(q, k, self.aux_embedding_dim)
            # q, k = self.W_q(att_emb).repeat_interleave(F, dim=1), self.W_k(att_emb).repeat(1, F, 1)
            # attn = self.att_gate(q * k).reshape(B, F, F)
            right_emb = X_i.repeat(1, F, 1)
            right_emb = torch.einsum('bfd,fde->bfe', right_emb, self.interaction_weight)
            tmp = left_emb * right_emb
            tmp = tmp.reshape(B, F, F, D)
            tmp = tmp * attn.unsqueeze(-1)
            tmp = tmp.sum(-2) * self.num_field # Alert
            if self.cnt % 4000 == 0:
                print(attn)
                torch.save(attn, 'attn_map_base.pth')
            X_i = X_i + tmp
            if self.analyzing:
                self.record_cross_emb.append(tmp.detach().clone().cpu())
                self.record_cross_emb_residual.append(X_i.detach().clone().cpu())
        return X_i.flatten(1)

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = torch.einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class GAU(nn.Module):
    def __init__(self, input_dim, num_layers, feature_map=None):
        super(GAU, self).__init__()
        self.num_layers = num_layers
        self.dim_emb = input_dim
        self.dropout = nn.Dropout(0.1)
        self.dim_att_emb = self.dim_emb
        self.norm = nn.LayerNorm(self.dim_att_emb)
        # self.att_embedding_layer = FeatureEmbedding(feature_map, self.dim_att_emb * 10)
        query_key_dim = self.dim_att_emb
        self.to_q = nn.Sequential(
            nn.Linear(self.dim_att_emb, query_key_dim),
            nn.SiLU()
        )
        self.to_k = nn.Sequential(
            nn.Linear(self.dim_att_emb, query_key_dim),
            nn.SiLU()
        )
        # self.to_hidden = nn.Sequential(
        #     nn.Linear(input_dim, input_dim * 2),
        #     nn.SiLU()
        # )
        # self.to_out = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.Dropout(0)
        # )
        self.offsetscale = OffsetScale(query_key_dim, heads = 2)
        self.cnt = 0

    def forward(self, X_0, X):
        # att_emb = self.att_embedding_layer(X, flatten_emb=True)
        # att_emb = att_emb.reshape(att_emb.shape[0], -1, self.dim_att_emb)
        
        X_0 = X_0.reshape(X_0.shape[0], -1, self.dim_emb)
        seqlen = X_0.shape[-2]
        X_i = X_0 # b x dim
        self.cnt += 1
        for i in range(self.num_layers):
            # normed_x, residual = self.norm(X_i), X_i
            # v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)
            v, normed_x, gate, residual = X_i, self.norm(X_i), X_0, X_i

            # qk = self.to_qk(normed_x)
            # q, k = self.offsetscale(qk)
            q, k = self.to_q(X_0), self.to_k(X_i)
            
            sim = torch.einsum('b i d, b j d -> b i j', q, k)
            # attn = torch.relu(sim / seqlen) ** 2
            # attn = (sim / 0.01).softmax(-1)
            attn = nn.functional.normalize(sim, dim=-1)
            # attn = torch.tanh(sim) * 1000
            # attn = attn * 100
            attn = self.dropout(attn)
            if self.cnt % 1000 == 0:
                print(attn)
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = out * gate
            # out = nn.functional.normalize(out, dim=-1)
            X_i = out + residual
            # X_i = self.to_out(out) + residual
        return X_i.flatten(start_dim=1)

class CrossNetMix(nn.Module):
    """ CrossNetMix improves CrossNetV2 by:
        1. add MOE to learn feature interactions in different subspaces
        2. add nonlinear transformations in low-dimensional space
    """
    def __init__(self, in_features, layer_num=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for i in range(self.layer_num)])
        # V: (in_features, low_rank)
        self.V_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for i in range(self.layer_num)])
        # C: (low_rank, low_rank)
        self.C_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, low_rank, low_rank))) for i in range(self.layer_num)])
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(
            torch.empty(in_features, 1))) for i in range(self.layer_num)])
        # self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l
