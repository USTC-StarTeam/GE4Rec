

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
        self.kwargs = kwargs
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_feature_emb = []
        self.record_bi_pooling_vec = []
        self.record_gen = []
        self.record_gen_linear = []
        self.record_final_representation = []
        self.record_dnn_out = []

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        self.grad_var_list = []
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        if self.training and self.analyzing:
            feature_emb.retain_grad()
        self.feature_embedding_grad = feature_emb
        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())

        gen, _ = self.gen(feature_emb, X)
        if self.analyzing:
            self.record_gen.append(gen.detach().clone().cpu())

        row, col = torch.triu_indices(feature_emb.shape[1], feature_emb.shape[1], offset=1)
        if not self.symmetric:
            rst = gen[:, row] * feature_emb[:, col]
        else:
            rst = gen[:, row] * gen[:, col]
        bi_pooling_vec = rst.sum(-2)
        if self.analyzing:
            self.record_bi_pooling_vec.append(bi_pooling_vec.detach().clone().cpu())
            self.record_final_representation.append(bi_pooling_vec.detach().clone().cpu())
        if self.training and self.analyzing:
            bi_pooling_vec.retain_grad()
        self.grad_var_list.append(bi_pooling_vec)

        y_pred = self.lr_layer(X) + bi_pooling_vec.sum(dim=-1, keepdim=True)

        dnn_out = self.mlp(feature_emb.flatten(start_dim=1))
        y_pred += self.linear_out(dnn_out)
        
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict