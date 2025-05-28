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
import seaborn as sns

# plt.rcParams['font.size'] = 15
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

num_layers_dict = {
    'Criteo': {
        'DCNv2': 3,
    },
    'Avazu': {
        'DCNv2': 5,
    },
}

dataset_name = args.dataset_name
DATASET_ID = dataset_id_dict[dataset_name] # criteo_x1_7b681156, frappe_x1_04e961e9, avazu_x4_3bbbc4c9, criteo_x4_9ea3bdfc


MODEL_NAME = args.model_name
suffix_data = '_avazu' if dataset_name == 'Avazu' else '_criteo'
# suffix = [suffix_data + '', suffix_data + '_nonlinear_x0']
suffix = [suffix_data + '', suffix_data + '_gen']
print(f'Ploting {MODEL_NAME} on {dataset_name}!!!')
printed_name = suffix
printed_name = [MODEL_NAME + _ for _ in printed_name]

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


def count_files_in_folder(folder_path):
    file_count = 0
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    return file_count
emb_list = []
grad_list = []
emb_list_all = []

for idx, model_name in enumerate(model_name_list):
    emb_dir = os.path.join(f'figure/raw_data/{DATASET_ID}/{model_name}/', "emb")
    file_count = count_files_in_folder(emb_dir)
    emb_list_all.append([])
    with open(os.path.join(emb_dir, f'emb_epoch0.pth'), 'rb') as handle:
        emb_dict = pickle.load(handle)
        if 'grad' in emb_dict:
            grad_list.append(emb_dict.pop('grad'))
        emb_dict = {k: v.cpu() for k, v in emb_dict.items()}
        emb_list_all[-1].append(emb_dict)
        emb_list.append(emb_dict)


def get_cov(data):
    m = data.mean(0, keepdim=True)
    rst = (data - m).T @ (data - m) / len(data)
    return rst

num_layers = num_layers_dict[dataset_name].get(MODEL_NAME, 1)

for idx, emb in enumerate(emb_list):
    if MODEL_NAME == 'DCNv2':
        A = emb['left'].flatten(1).tensor_split(num_layers)[0]
        B = emb['right'].flatten(1).tensor_split(num_layers)[0]
    elif MODEL_NAME == 'xDeepFM':
        A = emb['feature_emb'].flatten(1).tensor_split(num_layers)[0]
        B = emb['gating'].flatten(1).tensor_split(num_layers)[0]
    else:
        A = emb['feature_emb'].flatten(1).tensor_split(num_layers)[0]
        B = emb['gating'].flatten(1).tensor_split(num_layers)[0]

    print(A.shape, B.shape)
    A = (A - A.mean(0)) / A.std(0)
    B = (B - B.mean(0)) / B.std(0)
    cov = (A.T @ B / (A.shape[0] - 1))
    torch.nan_to_num_(cov, nan=0)
    cov = cov
    plt.figure(figsize=(5, 4), dpi=400)
    sns.heatmap(cov, annot=False, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, vmin=-1, vmax=1)
    plt.title('Pearson Correlation between fields')
    plt.xlabel('Index')
    plt.ylabel('Index')

    save_dir = f'./{plot_dir}/final/cov/cov_matrix/{dataset_name}/{MODEL_NAME}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{printed_name[idx]}_{0}.jpg'))
