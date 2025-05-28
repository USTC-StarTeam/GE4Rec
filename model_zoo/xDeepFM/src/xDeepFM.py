


import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, LogisticRegression
from fuxictr.pytorch import GEN

class xDeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="xDeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 dnn_hidden_units=[64, 64, 64], 
                 dnn_activations="ReLU",
                 cin_hidden_units=[16, 16, 16], 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 **kwargs):
        super(xDeepFM, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      gpu=gpu, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer,
                                      **kwargs)     
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'prelu': nn.PReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'linear': nn.Identity(),
        }
        self.concat_emb = kwargs["concat_emb"]
        self.gamma = kwargs["gamma"]
        self.symmetric = kwargs["symmetric"]
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1, 
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm) \
                   if dnn_hidden_units else None # in case of only CIN used
        self.lr_layer = LogisticRegression(feature_map, use_bias=False)
        self.cin = CompressedInteractionNet(feature_map.num_fields, cin_hidden_units, output_dim=1, embedding_dim=embedding_dim, feature_map=feature_map, kwargs=kwargs)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_feature_emb = []
        self.record_gen = []
        self.record_gen_linear = []
        # self.record_final_representation = []

    def forward(self, inputs):
        self.grad_var_list = []
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X) # list of b x embedding_dim
        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())
        if self.training and self.analyzing:
            feature_emb.retain_grad()
        self.feature_embedding_grad = feature_emb

        lr_logit = self.lr_layer(X)
        cin_logit = self.cin(feature_emb, feature_emb)
    
        y_pred = lr_logit + cin_logit # only LR + CIN
        if self.dnn is not None:
            dnn_logit = self.dnn(feature_emb.flatten(start_dim=1))
            y_pred += dnn_logit # LR + CIN + DNN
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class CompressedInteractionNet(nn.Module):
    def __init__(self, num_fields, cin_hidden_units, output_dim=1, embedding_dim=None, feature_map=None, kwargs=None):
        super(CompressedInteractionNet, self).__init__()
        self.cin_hidden_units = cin_hidden_units
        self.fc = nn.Linear(sum(cin_hidden_units), output_dim)
        self.cin_layer = nn.ModuleDict()
        self.gen_layers = nn.ModuleList()
        for i, unit in enumerate(self.cin_hidden_units):
            in_channels = num_fields * self.cin_hidden_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1) # kernel output shape
            self.gen_layers.append(GEN(
                feature_map,
                embedding_dim,
                # out_field=num_fields * embedding_dim,
                **kwargs
            ))

        self.num_layers = len(self.cin_hidden_units)
        self.embedding_dim = embedding_dim
        self.num_field = num_fields
        self.small_transformation = kwargs.get("small_transformation", False)
            
    def init_record(self):
        self.record_final_representation = []
        self.record_X_0 = []

    def forward(self, feature_emb, mask=None):
        pooling_outputs = []
        X_0 = feature_emb
        batch_size = feature_emb.shape[0]
        embedding_dim = feature_emb.shape[-1]
        X_i = feature_emb
        for i in range(len(self.cin_hidden_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", self.gen_layers[i](X_0)[0], X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        if self.analyzing:
            self.record_final_representation.append(torch.cat(pooling_outputs, dim=-1).detach().clone().cpu())
        output = self.fc(torch.cat(pooling_outputs, dim=-1))
        return output
        