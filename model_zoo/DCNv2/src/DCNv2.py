

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

        self.fc = nn.Sequential(
            nn.Linear(final_dim, final_dim * 2),
            nn.ReLU(),
            nn.Linear(final_dim * 2, 1),
        )
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_feature_emb = []
        self.record_final_out = []
        self.record_feature_emb = []
        self.record_gen = []
        self.record_gen_linear = []
        self.record_final_representation = []

    def forward(self, inputs):
        self.grad_var_list = []
        X = self.get_inputs(inputs)

        feature_emb = self.embedding_layer(X, flatten_emb=True)

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

    def init_record(self):
        self.record_cross_emb = []
        self.record_cross_emb_residual = []
        self.record_non_linear_rep = []
        self.record_cross_rst = []
        self.record_X_0 = []
        self.record_left = []
        self.record_right = []

    def forward(self, feature_embedding, mask=None):
        X_0 = feature_embedding
        X_i = feature_embedding # b x dim

        for i in range(self.num_layers):
            if self.kwargs.get('use_gen_rst', False):
                X_0 = self.gen_layers[i](X_i)[0]
            else:
                X_0 = self.gen_layers[i](feature_embedding)[0]
            tmp = self.cross_layers[i](X_i)

            if self.analyzing:
                self.record_X_0.append(X_0.detach().clone().cpu())
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