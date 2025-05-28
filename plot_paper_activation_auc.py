#!/usr/bin/env python
# coding: utf-8

import os
import glob
import math
import json
import torch
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

plt.rcParams['font.size'] = 18
# colors = ["#89c8e8","#caa2f4","#1f78b4","#6a3d9a","#8acc72", "#ebd57c", "#ed4437", "#fc9a9a"]
colors = ["#fbb463","#80b1d3","#f47f72","#bdbadb","#fbf8b4", "#8dd1c6"]
ls_list = ['-', '--']
marker_list = ['o', 's']

# colors = ['#66BC98', '#AAD09D', '#E3EA96', '#FCDC89', '#E26844', '#8A233F', '#C2ABC8']
colormap = 'viridis'  # 可以替换为 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'Spectral', 'cubehelix'
# plt.rcParams['text.usetex'] = True

parser = ArgumentParser()
parser.add_argument('--model_name', type=str, default='DCNv2')
parser.add_argument('--dataset_name', type=str, default='Avazu')
args = parser.parse_args()


def get_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def count_files_in_folder(folder_path):
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

plot_dir = 'figure'

dataset_id_dict = {
    'Criteo': 'criteo_x1_7b681156',
    'Avazu': 'avazu_x4_3bbbc4c9',
}

dataset_name = args.dataset_name
DATASET_ID = dataset_id_dict[dataset_name] # criteo_x1_7b681156, frappe_x1_04e961e9, avazu_x4_3bbbc4c9, criteo_x4_9ea3bdfc

activations = ['sigmoid', 'tanh', 'silu', 'relu']

MODEL_NAME = args.model_name
suffix_data = '_avazu' if dataset_name == 'Avazu' else '_criteo'
suffix = [suffix_data + '', suffix_data + '_gen'] + [suffix_data + f'_{act}' for act in activations[1:]]

MODEL_NAME = args.model_name

print(f'Ploting {MODEL_NAME} on {dataset_name}!!!')
printed_name = suffix
printed_name = [MODEL_NAME + _ for _ in printed_name]

printed_name = ['DIS', 'Linear', 'Sigmoid', 'Tanh', 'SiLU', 'ReLU']

model_name_list = [MODEL_NAME + _ for _ in suffix]

plot_tag = ''

early_stop_patience = 0

rst_dict = {}

num_field_dict = {
    'Criteo': 39, 'Avazu': 24,
}
num_field = num_field_dict[dataset_name]

def get_cardinality():
    path = os.path.join('data', dataset_name, DATASET_ID, 'feature_map.json')
    feature_map = json.load(open(path, 'r'))
    cardinality_list = []
    for _ in feature_map['features']:
        cardinality_list.append(list(_.values())[0].get('vocab_size', 0))
    return torch.tensor(cardinality_list)

def get_num_field(dataset_name):
    num_field = num_field_dict[dataset_name]
    row, col = torch.triu_indices(num_field, num_field, offset=1)
    return num_field, row, col
    

log_data_split_dict = {
    'train': 0,
    'valid': 1,
    'test': 2,
}
phase = 'valid'
chunk_size = 20000
log_data_split = log_data_split_dict[phase]


# ========================= AUC results ===========================

baseline_auc_dict = {
    'Avazu': {
        'FM': [0.788225, 0.792494],
        'DeepFM': [0.792790, 0.793334],
        'CrossNet': [0.788106, 0.79236],
        'DCNv2': [0.792587, 0.793512],
    },
    'Criteo': {
        'FM': [0.80236, 0.81108],
        'DeepFM': [0.81380, 0.81396],
        'CrossNet': [0.81177, 0.813687],
        'DCNv2': [0.813575, 0.814115],
    }
}
group_auc_dict = {
    'Avazu': {
        'FM': [0.790176, 0.791232, 0.792410],
        'DeepFM': [0.793062, 0.793006, 0.793215],
        'CrossNet': [0.790267, 0.791170, 0.792028],
        'DCNv2': [0.792318, 0.792704, 0.793038, 0.793610],
    },
    'Criteo': {
        'FM': [0.807131, 0.806604, 0.807103, 0.810787],
        'DeepFM': [0.813949, 0.813814, 0.813856, 0.814080],
        'CrossNet': [0.813017, 0.812832, 0.813176, 0.813723],
        'DCNv2': [0.813557, 0.813790, 0.813741, 0.814293],
    }
}
fig, ax1 = plt.subplots()
auc_list = [baseline_auc_dict[dataset_name][MODEL_NAME][0]] + group_auc_dict[dataset_name][MODEL_NAME] + [baseline_auc_dict[dataset_name][MODEL_NAME][1]]
low_line, high_line = baseline_auc_dict[dataset_name][MODEL_NAME]
for idx, auc in enumerate(auc_list):
    ax1.bar(printed_name[idx], auc, color=colors[idx], label=printed_name[idx])

# ax1.axhline(y=low_line, color='black', linestyle='--')
# ax1.axhline(y=high_line, color='black', linestyle='--')
ax1.set_ylabel('AUC')
plt.xticks([])
ax1.set_ylim(low_line * 0.999, high_line * 1.001)
ax1.set_xlabel('Different model variants')

ax1.spines['right'].set_visible(False)  # 隐藏右边框
ax1.spines['top'].set_visible(False)    # 隐藏上边框
# plt.axhline(y=auc_list[0], color=colors[0], linestyle='--')


plt.legend(fontsize=12)


save_dir = f'./figure/final/activation/auc/{dataset_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{MODEL_NAME}{plot_tag}.pdf'))
