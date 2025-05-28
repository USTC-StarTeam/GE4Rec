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

activations = ['sigmoid', 'relu', 'tanh', 'silu', 'elu']

MODEL_NAME = args.model_name
suffix = ['_avazu', '_avazu_gen', '_criteo', '_criteo_gen']
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
    DATASET_ID = dataset_id_dict['Avazu'] if idx < 2 else dataset_id_dict['Criteo']
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


# # Spectrum

def singular_value_sum(S):
    return S.sum(-1)

def information_abundance(S):
    return S.sum(-1) / S.max(-1).values

def get_cov(data):
    m = data.mean(0, keepdim=True)
    rst = (data - m).T @ (data - m) / len(data)
    return rst

def RankMe(Z, epsilon=1.0e-7):
    # Z: [..., B, D]
    S = torch.svd(Z).S
    p_vector = S / S.sum(-1, keepdim=True) + epsilon
    rst = torch.exp((- p_vector * torch.log(p_vector)).sum(-1))
    return rst

def get_cut_point(S):
    prev = 9999
    cnt = 0
    for idx, _ in enumerate(S):
        if prev == _:
            cnt += 1
        prev = _
        if cnt == 2:
            return (idx, _)
    return (idx, _)

def plot_cut_point(ax, cut_point_list, y_len):
    for idx in range(0, 4, 2):
        left, S_left = cut_point_list[idx]
        right, S_right = cut_point_list[idx + 1]
        
        top, down = min(S_left, S_right) - 0.2, min(S_left, S_right) - 3
        
        ax.plot([left, left], [down, top], color=colors[idx // 2], ls=':')
        ax.plot([right, right], [down, top], color=colors[idx // 2], ls=':')

        pos = down + (top - down) * 0.75 if idx == 0 else down + (top - down) * 0.25
        arrowstyle = 'fancy'
        ax.annotate('', xy=(right, pos), xytext=(left, pos),
                    arrowprops=dict(arrowstyle=arrowstyle, color=colors[idx // 2], lw=1))

def fill_gap(ax, x, y_low, y_high, color):
    # 填充DeepFM高于FM的区域
    step = len(x) // 10
    ax.fill_between(x, y_low, y_high, where=(np.array(y_high) > np.array(y_low)), 
                    interpolate=True, color=color, alpha=0.2)

    # 添加向上的箭头
    for i in range(step, len(x), step):
        if y_high[i] > y_low[i]:
            if y_high[i] > y_low[i] + 0.7 and y_high[i - step // 4] > y_low[i - step // 4] + 0.5:
                ax.annotate('', xy=(x[i], y_high[i]), xytext=(x[i], y_low[i]),
                            arrowprops=dict(arrowstyle='->, head_width=0.15', color=color, lw=1, shrinkA=0.3, shrinkB=0.3))

fig, ax1 = plt.subplots()
y_list = []

# Direct embedding
variant = ['DIS', 'GEN']

cut_point_list = []
y_list = []
for idx, emb in enumerate(emb_list):
    model_name = printed_name[idx]
    d_name = 'Avazu' if idx < 2 else 'Criteo'
    num_field, row, col = get_num_field(d_name)
    if 'DCNv2' in model_name:
        y = emb['X_0'].flatten(1)
    elif 'xDeepFM' in model_name:
        if idx < 3:
            y = emb['gating'].flatten(1)
        elif idx == 3:
            y = emb['X_0'].flatten(1)
    else:
        y = emb['gating'].flatten(1)

    y = get_cov(y)
    y = torch.svd(y).S
    y = y / y.max()
    y = y.log()
    y_list.append(y)
    x = list(range(len(y)))
    ax1.plot(x, y, label=f'{d_name}-{variant[idx % 2]}', c=colors[idx // 2], ls=ls_list[idx % 2], lw=2)
    
    cut_point_list.append(get_cut_point(y))
plot_cut_point(ax1, cut_point_list, len(y))
fill_gap(ax1, list(range(len(y_list[0]))), y_list[0], y_list[1], colors[0])
fill_gap(ax1, list(range(len(y_list[2]))), y_list[2], y_list[3], colors[1])
ax1.legend()

ax1.spines['right'].set_visible(False)  # 隐藏右边框
ax1.spines['top'].set_visible(False)    # 隐藏上边框
ax1.set_xlabel('Singular Value Rank Index')
ax1.set_ylabel('Logarithm of Normalized Singular Value', fontdict={'size': 14})

save_dir = f'./{plot_dir}/final/direct_embedding/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{MODEL_NAME}{plot_tag}.pdf'))
