#!/bin/bash
echo "Training recommendation models"
model_name=DeepFM

python model_zoo/${model_name}/run_expid.py --expid ${model_name}_avazu --gpu 3
python model_zoo/${model_name}/run_expid.py --expid ${model_name}_avazu_fs --gpu 3
# python model_zoo/${model_name}/run_expid.py --expid ${model_name}_criteo --gpu 0
# python model_zoo/${model_name}/run_expid.py --expid ${model_name}_criteo_fs --gpu 0
