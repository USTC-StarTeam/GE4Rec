# Quickstart

This repo is the code of ICML2025 submission "From Feature Interaction to Feature Generation: A Generative Paradigm of CTR Prediction Models".
This repo is implemented based on [FuxiCTR](https://github.com/reczoo/FuxiCTR).

## Enviroments

```bash
conda create -n FuxiCTR_analysis python=3.10 -y
conda activate FuxiCTR_analysis
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

## Prepare dataset

```bash
bash 1.prepare.sh
```

## Reproduce experiments

The following script will reproduce results on DCN V2
```bash
bash 2.reproduce.sh
bash 3.analyze.sh
```


## Advanced usage & Code explanations

### Faster training with preprocessed data (Highly Recommended!!!)

   After the first run, FuxiCTR generates the `parquet` format dataset (that can be found in `data/Avazu/avazu_x4_3bbbc4c9`). You should change the following entries of dataset config files for faster training. For example, in `model_zoo/FM/config/dataset_config.yaml`, and similarly for other models, make these changes:
   ```yaml
   avazu_x4_3bbbc4c9:
      data_format: parquet # original: csv
      ...
      ...
      rebuild_dataset: false # original: true
      test_data: ../../data/Avazu/avazu_x4_3bbbc4c9/test.parquet # original: test.csv
      train_data: ../../data/Avazu/avazu_x4_3bbbc4c9/train.parquet # original: train.csv
      valid_data: ../../data/Avazu/avazu_x4_3bbbc4c9/valid.parquet # original: valid.csv
   ```

### Generate other embeddings for analysis

After experiments, we can perform model inference based on the saved checkpoints (e.g., `model_zoo/DeepFM/Avazu/DeepFM_avazu_x4_001/avazu_x4_3bbbc4c9/`).

1. We should register embeddings that need to be saved in a `init_record` function (please refer to `model_zoo/DCNv2/src/DCNv2.py`). This should follow a `record_XXX` format, where `XXX` is the name of embeddings that you want to save for future analysis. The following line will save the feature embeddings. Remarkably, embeddings required for analysis in the paper have already been registered.

```python
def init_record(self):
   self.record_feature_emb = []
   ...
```

2. We should change the forward function to record the specified embedding, like:
   
```python
def forward(self, inputs):
   X = self.get_inputs(inputs)
   feature_emb = self.embedding_layer(X, flatten_emb=True)
   if self.analyzing:
      self.record_feature_emb.append(
         feature_emb.detach().clone().cpu()
      )
   ...
```
