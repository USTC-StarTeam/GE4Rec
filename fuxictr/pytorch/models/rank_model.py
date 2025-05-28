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
import random
import pickle
import torch.nn as nn
import numpy as np
import torch
import os, sys
import logging
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from fuxictr.utils import Monitor, not_in_whitelist
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import math
from fuxictr.pytorch.dataloaders.npz_block_dataloader import NpzBlockDataLoader
from fuxictr.pytorch.dataloaders.npz_dataloader import NpzDataLoader
from fuxictr.pytorch.dataloaders.parquet_block_dataloader import ParquetBlockDataLoader
from fuxictr.pytorch.dataloaders.parquet_dataloader import ParquetDataLoader

class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 task="binary_classification", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 early_stop_patience=2, 
                 eval_steps=None, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.model_name = kwargs['model']
        self.dataset_id = kwargs['dataset_id']
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]
        self.init_analyzer(feature_map, kwargs['data_format'], kwargs['train_data'], kwargs["valid_data"], kwargs["test_data"])

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            if type(module) == nn.Embedding:
                                if self._embedding_regularizer:
                                    for emb_p, emb_lambda in emb_reg:
                                        reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                            else:
                                if self._net_regularizer:
                                    for net_p, net_lambda in net_reg:
                                        reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        torch.nn.functional.binary_cross_entropy(return_dict["y_pred"], y_true)
        loss += self.regularization_loss()
        return loss

    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            if hasattr(m, 'reset_custom_params'):
                m.reset_custom_params()
        self.apply(reset_default_params)
        self.apply(reset_custom_params)

    def get_inputs(self, inputs, feature_source=None):
        X_dict = dict()
        for feature in inputs.keys():
            if feature in self.feature_map.labels:
                continue
            spec = self.feature_map.features[feature]
            if spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(spec["source"], feature_source):
                continue
            X_dict[feature] = inputs[feature].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[self.feature_map.group_id]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr

    def init_analyzer(self, feature_map, data_format, train_data, valid_data, test_data):
        self.analyzing = False
        def generate_analyze_dataloader(feature_map, data_format, data):
            if data_format == "npz":
                DataLoader = NpzDataLoader
            else: # ["parquet", "csv"]
                DataLoader = ParquetDataLoader
            return DataLoader(feature_map, data, split="train", batch_size=50000,
                                   shuffle=True, num_workers=1)
        
        self.analyzer_root_path = os.path.join('figure', 'raw_data', self.dataset_id, self.model_id)
        if not os.path.exists(self.analyzer_root_path):
            os.makedirs(self.analyzer_root_path)
        self.robustness_step = []
        self.batch_data = []
        # robustness results
        # discriminability results
        self.spectrum_list = defaultdict(list)
        self.analyze_loader_list = []
        # self.analyze_loader_list.append(generate_analyze_dataloader(feature_map, data_format, train_data))
        self.analyze_loader_list.append(generate_analyze_dataloader(feature_map, data_format, valid_data))
        # self.analyze_loader_list.append(generate_analyze_dataloader(feature_map, data_format, test_data))
        
        # def get_X_y(dataloader):
        #     dataset = dataloader.dataset.darray
        #     return dataset[:, :-1], dataset[:, -1]
        # rst = [get_X_y(_) for _ in self.analyze_loader_list]
        # X, y = [_[0] for _ in rst], [_[1] for _ in rst]
        # with open(os.path.join(self.analyzer_root_path, 'batch_data.pth'), 'wb+') as handle:
        #     pickle.dump([X, y], handle)
        
        self.grad_list = []

    def analyze(self):
        self.analyze_start()
        self.analyze_epoch_start()
        self.analyze_epoch()
        self.analyze_epoch_end()
        self.analyze_end()

    def analyze_start(self, max_gradient_norm=10):
        self.log_batch_data = True
        self.set_analyze_state('end')
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        return

    def analyze_epoch_start(self):
        self.set_analyze_state('start')

    def set_analyze_state(self, mode=None):
        if mode == 'start':
            for module in self.modules():
                setattr(module, 'analyzing', True)
                if hasattr(module, "init_record"):
                    method = getattr(module, "init_record")
                    method()
        elif mode == 'end':
            for module in self.modules():
                setattr(module, 'analyzing', False)
        else:
            raise NotImplementedError

    def analyze_epoch(self):
        def save_random_state():
            random_state = {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'python': random.getstate()
            }
            return random_state
        def load_random_state(random_state):
            torch.set_rng_state(random_state['torch'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(random_state['cuda'])
            np.random.set_state(random_state['numpy'])
            random.setstate(random_state['python'])
        original_random_state = save_random_state()
        self.grad_var_list = []
        for idx, loader in enumerate(self.analyze_loader_list):
            if idx == 0:
                self.train()
            else:
                self.eval()
            torch.manual_seed(2025)
            batch_data = next(iter(loader))
            if self.log_batch_data:
                batch_data_cpu = {k: v.detach().cpu() for k, v in batch_data.items()}
                self.batch_data.append(batch_data_cpu)
            self.analyze_step(batch_data, training=(idx == 0))
        
        load_random_state(original_random_state)

    def analyze_step(self, batch_data, training=False):
        if not training:
            self.forward(batch_data)
        else:
            self.optimizer.zero_grad()
            return_dict = self.forward(batch_data)
            y_true = self.get_labels(batch_data)
            loss = self.compute_loss(return_dict, y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            # for param_name, param in self.named_parameters():
            #     if param.grad is not None:
            #         self.grad_list.append((param_name, param.grad.detach().cpu()))
            # self.grad_list.append(('feature_embedding_grad', self.feature_embedding_grad.grad.detach().cpu()))
            # self.grad_list.append(('ecn_H_grad', [_.grad.detach().cpu() for _ in self.ecn_H_grad_list]))
            # self.grad_list.append(('lcn_H_grad', [_.grad.detach().cpu() for _ in self.lcn_H_grad_list]))
            for idx, grad in enumerate(self.grad_var_list):
                self.grad_list.append((f'grad_var_{idx}', grad.grad.detach().cpu()))
        self.eval()

    def analyze_epoch_end(self, prefix=''):
        logging.info(f'Start Analyzing')
        def get_recorded_emb():
            log_attributes_values = {}
            for module in self.modules():
                all_members = vars(module)
                for member_name, member in all_members.items():
                    if member_name.startswith("record"):
                        log_attributes_values[member_name[len("record_"):]] = member
            log_attributes_values = {k: torch.cat(v) for k, v in log_attributes_values.items() if len(v) > 0}
            log_attributes_values['grad'] = self.grad_list
            emb_root_path = os.path.join(self.analyzer_root_path, 'emb')
            if not os.path.exists(emb_root_path):
                os.makedirs(emb_root_path)
            with open(os.path.join(emb_root_path, f'{prefix}emb_epoch{self._epoch_index}.pth'), 'wb+') as handle:
            # with open(os.path.join(emb_root_path, f'emb_epoch0{suffix}.pth'), 'wb+') as handle:
                pickle.dump(log_attributes_values, handle)
            return log_attributes_values

        get_recorded_emb()
        # get_spectrum(recorded_emb)
        if self.log_batch_data:
            with open(os.path.join(self.analyzer_root_path, 'batch_data.pth'), 'wb+') as handle:
                pickle.dump(self.batch_data, handle)
            self.log_batch_data = False
        self.set_analyze_state('end')

    def analyze_end(self):
        def plot_spectrum():
            for name, spectrum_list in self.spectrum_list.items():
                n_epoch = len(spectrum_list)
                spectrum_save_root_path = os.path.join(self.analyzer_root_path, 'spectrum')
                if not os.path.exists(spectrum_save_root_path):
                    os.makedirs(spectrum_save_root_path)
                if len(spectrum_list[0].shape) == 1:
                    # Sample level, plot one graph
                    fig = plt.figure(dpi=1000)
                    x = list(range(n_epoch))
                    y = []
                    for idx, spectrum in enumerate(spectrum_list):
                        # y.append((spectrum.sum()).cpu().tolist())
                        y.append((spectrum.sum() / spectrum.max()).cpu().tolist())
                    plt.xlabel('Epoch')
                    plt.ylabel('SVS')
                    plt.plot(x, y)
                    with open(os.path.join(spectrum_save_root_path, name + '-sum.pth'), 'wb+') as handle:
                        pickle.dump([x, y], handle)
                    # torch.save([x, y], os.path.join(spectrum_save_root_path, name + '-sum.pth'))
                    plt.savefig(os.path.join(spectrum_save_root_path, name + '-sum.pdf'))
                elif len(spectrum_list[0].shape) == 2:
                    n_field = len(spectrum_list[0])
                    fig, axes = plt.subplots(math.ceil(n_field / 5), 5, figsize=(5 * 2, math.ceil(n_field / 5) * 2), dpi=1000)
                    x = list(range(n_epoch))
                    # y = [(_.sum(-1)).cpu().tolist() for _ in spectrum_list]
                    y = [(_.sum(-1) / _.max(-1).values).cpu().tolist() for _ in spectrum_list]
                    for i, ax in enumerate(axes.flat):
                        if i >= n_field:
                            ax.axis('off')
                            continue
                        ax.plot(x, [_[i] for _ in y])
                        # 设置x轴和y轴的标签
                        ax.set_xlabel(f'Field{i}')
                        ax.set_ylabel('SVS')
                    plt.tight_layout()
                    with open(os.path.join(spectrum_save_root_path, name + '-sum.pth'), 'wb+') as handle:
                        pickle.dump([x, y], handle)
                    # torch.save([x, y], os.path.join(spectrum_save_root_path, name + '-sum.pth'))
                    plt.savefig(os.path.join(spectrum_save_root_path, name + '-sum.pdf'))
                    pass
            pass
        # plot_spectrum()
        # with open(os.path.join(self.analyzer_root_path, 'batch_data.pth'), 'wb+') as handle:
        #     pickle.dump(self.batch_data, handle)
        # torch.save(self.batch_data, os.path.join(self.analyzer_root_path, 'batch_data.pth'))
        return

    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch
        
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        self.analyze_start()
        
        # self.analyze_epoch_start()
        # self.analyze_epoch()
        # self.analyze_epoch_end(prefix='per-epoch-')
        
        for epoch in range(epochs):
            self._epoch_index = epoch + 1
            self.train_epoch(data_generator)
            
            # self.analyze_epoch_start()
            # self.analyze_epoch()
            # self.analyze_epoch_end(prefix='per-epoch-')
            
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)
        self.analyze_end()

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                train_loss = 0
                self.eval_step()

            # watch_list = [1, 10, 100, 500, 1000, 2000, 3000]
            # if batch_index in watch_list:
            #     self.analyze_epoch_start()
            #     self.analyze_epoch()
            #     self.analyze_epoch_end(suffix='_' + str(batch_index))
            #     if batch_index == watch_list[-1]:
            #         break

            if self._stop_training:
                break

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        miss_para = self.load_state_dict(state_dict, strict=True)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))
