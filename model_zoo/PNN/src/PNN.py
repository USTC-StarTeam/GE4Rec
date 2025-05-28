

from torch import nn
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InnerProductInteraction
from fuxictr.pytorch import GEN


class PNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="PNN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 product_type="inner", 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(PNN, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs) 
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        
        self.symmetric = kwargs["symmetric"]
        self.gen = GEN(feature_map, embedding_dim, **kwargs)
        
        if product_type != "inner":
            raise NotImplementedError("product_type={} has not been implemented.".format(product_type))
        self.inner_product_layer = InnerProductInteraction(feature_map.num_fields, output="inner_product")
        input_dim = int(feature_map.num_fields * (feature_map.num_fields - 1) / 2) \
                  + feature_map.num_fields * embedding_dim
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) 
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_feature_emb = []
        self.record_gen = []
        self.record_gen_linear = []
        self.record_final_representation = []

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

        gen = self.gen(feature_emb)[0]

        row, col = torch.triu_indices(feature_emb.shape[1], feature_emb.shape[1], offset=1)
        rst = gen[:, row] * feature_emb[:, col]
        inner_products = rst.sum(-1)
        
        dense_input = torch.cat([feature_emb.flatten(start_dim=1), inner_products], dim=1)
        
        if self.analyzing:
            self.record_final_representation.append(dense_input.detach().clone().cpu())

        y_pred = self.dnn(dense_input)
        return_dict = {"y_pred": y_pred}
        return return_dict