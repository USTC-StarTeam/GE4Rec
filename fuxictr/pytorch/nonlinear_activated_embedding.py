import torch
import torch.nn as nn
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, SqueezeExcitation, ScaledDotProductAttention

class FeatureSelection(nn.Module):
    def __init__(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[], 
                 fs1_context=[]):
        super(FeatureSelection, self).__init__()
        self.fs1_context = fs1_context
        
        # self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim)
        # self.fs1_gate = MLP_Block(input_dim=feature_dim,
        #                           output_dim=feature_dim,
        #                           hidden_units=fs_hidden_units,
        #                           hidden_activations="ReLU",
        #                           output_activation="Sigmoid",
        #                           batch_norm=False)
        
        if len(fs1_context) == 0:
            self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
                                                required_feature_columns=fs1_context)
        self.fs1_gate = MLP_Block(input_dim=embedding_dim * max(1, len(fs1_context)),
                                  output_dim=feature_dim,
                                  hidden_units=fs_hidden_units,
                                  hidden_activations="ReLU",
                                  output_activation="Sigmoid",
                                  batch_norm=False)

    def forward(self, X, flat_emb):
        # fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
        
        if len(self.fs1_context) == 0:
            fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        else:
            fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
            
        # fs1_input = flat_emb
        
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1
        return feature1

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
        residual = X
        
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
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output

class GEN(nn.Module):
    def __init__(self, feature_map, embedding_dim, out_field=None, **kwargs) -> None:
        super().__init__()
        self.num_field = feature_map.num_fields if out_field is None else out_field
        self.embedding_dim = embedding_dim
        self.use_senet = kwargs.get('use_senet', False)
        self.use_finalmlp = kwargs.get('use_finalmlp', False)
        if self.use_senet:
            self.feature_gating = SqueezeExcitation(self.num_field, reduction_ratio=kwargs['reduction_ratio'], excitation_activation='ReLU')
        elif self.use_finalmlp:
            self.feature_gating = FeatureSelection(feature_map, feature_map.sum_emb_out_dim(), embedding_dim, kwargs['fs_hidden_units'], kwargs['fs1_context'])
        else:
            if kwargs.get('more_layers', False):
                self.feature_gating = nn.Sequential(
                    nn.Linear(self.num_field * embedding_dim, self.num_field * embedding_dim, bias=False),
                    nn.BatchNorm1d(self.num_field * embedding_dim),
                    nn.ReLU(),
                    nn.Linear(self.num_field * embedding_dim, self.num_field * embedding_dim, bias=False),
                )        
            elif kwargs.get('use_att', False):
                self.feature_gating = MultiHeadSelfAttention(
                    embedding_dim, embedding_dim, 4, 0.2, use_residual=False, use_scale=True, layer_norm=True,
                )
            else:
                self.feature_gating = nn.Sequential(
                    nn.Linear(self.num_field * embedding_dim, self.num_field * embedding_dim, bias=False),
                ) if kwargs["concat_emb"] else nn.Parameter(nn.init.xavier_normal_(torch.zeros(self.num_field, embedding_dim, embedding_dim)))
            # self.feature_gating = nn.Sequential(
            #     nn.Linear(feature_map.sum_emb_out_dim(), feature_map.sum_emb_out_dim(), bias=False),
            #     nn.ReLU(),
            #     nn.Linear(feature_map.sum_emb_out_dim(), feature_map.sum_emb_out_dim(), bias=False),
            # ) 

            # self.transform_matrix = nn.Parameter(torch.randn(feature_map.sum_emb_out_dim(), feature_map.sum_emb_out_dim()))
            # nn.init.xavier_normal_(self.transform_matrix)
            # self.transform_matrix_mask = self.get_transform_matrix_mask(self.num_field, self.embedding_dim)
            # self.feature_gating = nn.Sequential(
            #     nn.Linear(feature_map.sum_emb_out_dim(), embedding_dim, bias=False),
            #     nn.Linear(embedding_dim, feature_map.sum_emb_out_dim(), bias=False),
            # ) if kwargs["concat_emb"] else nn.Linear(embedding_dim, embedding_dim)
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
            self.symmetric = kwargs["symmetric"]
            self.small_transformation = kwargs.get("small_transformation", False)

            self.cardinality = self.get_cardinality(feature_map)
            self.set_cardinality_group()
            self.exp_group_idx = kwargs.get('exp_group_idx', None)
            self.batch_norm = nn.BatchNorm1d(1)
            self.use_bn = kwargs.get("use_bn", False)
            self.gamma = kwargs["gamma"]
            self.kwargs = kwargs
            self.get_exclude_mask()
            # self.gamma = nn.Parameter(torch.ones(1, dtype=torch.float))
            # self.gamma = nn.Parameter(torch.ones(1, dtype=torch.float) * kwargs["gamma"])

    def get_exclude_mask(self):
        row_indices = torch.arange(self.num_field * self.embedding_dim)
        col_indices = torch.arange(self.num_field * self.embedding_dim)

        row_block_indices = row_indices // self.embedding_dim
        col_block_indices = col_indices // self.embedding_dim

        is_on_diagonal_block = (row_block_indices.unsqueeze(1) == col_block_indices.unsqueeze(0))
        self.exclue_mask = torch.where(is_on_diagonal_block, torch.tensor(0), torch.tensor(1)).to(torch.bool)

    def get_transform_matrix_mask(self, num_field, embedding_dim):
        transform_matrix_mask = torch.ones(num_field * embedding_dim, num_field * embedding_dim)
        for i in range(num_field):
            transform_matrix_mask[i * embedding_dim:(i + 1) * embedding_dim, i * embedding_dim:(i + 1) * embedding_dim] = 0

        return transform_matrix_mask.bool()

    def get_cardinality(self, feature_map):
        tmp = list(feature_map.features.values())
        cardinality = torch.tensor([_.get('vocab_size', 0) for _ in tmp])
        idx = torch.sort(cardinality, descending=True).indices
        return idx

    def set_cardinality_group(self):
        if self.num_field == 24: # Avazu
            self.cardinality_group = [(0, 10), (14, 24)]
        elif self.num_field == 39: # Criteo
            self.cardinality_group = [(0, 10), (29, 39)]
        # if self.num_field == 24: # Avazu
        #     self.cardinality_group = [(0, 2), (2, 7), (7, 24)]
        # elif self.num_field == 39: # Criteo
        #     self.cardinality_group = [(0, 5), (5, 10), (10, 15), (15, 39)]

    def forward(self, feature_emb, X=None, mask=None):
        if self.use_senet:
            gating = self.feature_gating(feature_emb).reshape_as(feature_emb)
            gating_linear = gating
        elif self.use_finalmlp:
            gating = self.feature_gating(X, feature_emb.flatten(1)).reshape_as(feature_emb)
            gating_linear = feature_emb
        else:
            if self.concat_emb:
                if self.kwargs.get('field_share', False):
                    gating_linear = self.feature_gating(feature_emb.flatten(1)).reshape(-1, self.num_field, self.embedding_dim)[:, 0:1].repeat(1, self.num_field, 1).flatten(1)
                elif self.kwargs.get('use_att', False):
                    gating_linear = self.feature_gating(feature_emb.reshape(-1, self.num_field, self.embedding_dim)).flatten(1)
                else:
                    if self.kwargs.get('exclude_self', False):
                        mask = self.exclue_mask.to(feature_emb.device)
                        gating_linear = feature_emb.flatten(1) @ (self.feature_gating[0].weight.T * mask)
                    else:
                        gating_linear = self.feature_gating(feature_emb.flatten(1))
                gating_linear = gating_linear.reshape_as(feature_emb)

            else:
                if self.small_transformation:
                    gating_linear = torch.einsum('bfd,fde->bfe', feature_emb.reshape(-1, self.num_field, self.embedding_dim), self.feature_gating).flatten(1)
                else:
                    gating_linear = feature_emb
            if not self.use_bn:
                gating = self.nonlinear(gating_linear) * self.gamma
            else:
                gating = self.batch_norm(gating_linear.flatten(1).unsqueeze(1)).reshape_as(feature_emb)
                gating = self.nonlinear(gating) * self.gamma
            if self.exp_group_idx is not None:
                gating = gating.reshape(-1, self.num_field, self.embedding_dim)
                tmp = feature_emb.reshape(-1, self.num_field, self.embedding_dim) * 1

                left_idx = self.cardinality_group[self.exp_group_idx][0]
                right_idx = self.cardinality_group[self.exp_group_idx][1]
                tmp[:, self.cardinality[left_idx:right_idx]] = gating[:, self.cardinality[left_idx:right_idx]]
                gating = tmp

                gating = gating.reshape_as(feature_emb)
        if self.kwargs.get('gen_residual', False):
            gating = gating + feature_emb
        return gating, gating_linear
