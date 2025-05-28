# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
# sys.path.append('../../')
import logging
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
# import src
import gc
import argparse
import os
from pathlib import Path
import importlib.util


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='WuKong')
    parser.add_argument('--expid', type=str, default='WuKong_avazu', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    
    args['config'] = os.path.join('model_zoo', args['model'], 'config')
    model_src_path = f"model_zoo.{args['model']}.src"
    src = importlib.import_module(model_src_path)
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    for key, value in params.items():
        if key in ['data_root', 'test_data', 'train_data', 'valid_data']:
            params[key] = value[6:] # change to relative path
    params['gpu'] = args['gpu']
    # set_logger(params)
    # logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    # logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    model.checkpoint = os.path.join('model_zoo', args['model'], f"{params['model_root']}", params['dataset_id'], f"{experiment_id}.model")
    model.checkpoint = os.path.normpath(model.checkpoint)
    model.load_weights(model.checkpoint)
    model.analyze()
    
    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)
        print(experiment_id)
        print(test_result)
