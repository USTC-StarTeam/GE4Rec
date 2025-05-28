
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
        self.record_feature_emb_gen = []
        self.record_gen = []
        self.record_gen_linear = []
        self.record_final_representation = []
        self.record_left = []
        self.record_right = []

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        self.grad_var_list = []
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)

        self.cov_loss = 0

        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())
        if self.training and self.analyzing:
            feature_emb.retain_grad()
        self.feature_embedding_grad = feature_emb

        gen, gen_linear = self.gen(feature_emb)
        self.infonce_loss = 0
        # Code for analyzing
        if self.analyzing:
            self.record_gen.append(gen.detach().clone().cpu())
            self.record_gen_linear.append(gen_linear.detach().clone().cpu())
        if self.training and self.analyzing:
            gen.retain_grad()
            gen_linear.retain_grad()
        self.grad_var_list.append(gen)
        self.grad_var_list.append(gen_linear)

        row, col = torch.triu_indices(feature_emb.shape[1], feature_emb.shape[1], offset=1)
        left, right = gen[:, row], gen[:, col]
        if self.symmetric:
            left, right = gen[:, row], gen[:, col]
        else:
            left, right = gen[:, row], feature_emb[:, col]

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