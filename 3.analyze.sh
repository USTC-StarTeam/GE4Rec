#!/bin/bash
model_name=DeepFM

echo "Generating embeddings for analysis"

python analyze.py --model ${model_name} --expid ${model_name}_avazu --gpu 3
python analyze.py --model ${model_name} --expid ${model_name}_avazu_gen --gpu 3
python analyze.py --model ${model_name} --expid ${model_name}_criteo --gpu 0
python analyze.py --model ${model_name} --expid ${model_name}_criteo_gen --gpu 0

echo "Plot figures. (Note: ablation studies are only conducted for DCN V2 on Avazu)"

python plot_paper_collapse.py --model_name ${model_name}
python plot_paper_cov_matrix.py --model_name ${model_name} --dataset_name Avazu

# python plot_paper_ab_source_auc.py --model_name ${model_name} --dataset_name Avazu
# python plot_paper_ab_decoder_auc.py --model_name ${model_name} --dataset_name Avazu
# python plot_paper_ab_target_auc.py --model_name ${model_name} --dataset_name Avazu
