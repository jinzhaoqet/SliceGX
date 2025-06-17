## SliceGX
This repository contains the source code for our paper: SliceGX: Layer-wise GNN Explanation with Model-slicing.

Full version: [2025ICDE_SliceGX.pdf](https://github.com/TingtingZhu-ZJU/SliceGX/blob/main/Full_version.pdf)

## Requirements
- Pytorch 1.10.1
- PYG 2.1.0
- matplotlib, scipy, networkx, hydra, numpy ...

## datasets
- [Ba-shapes](https://github.com/divelab/DIG_storage/blob/main/xgraph/datasets/BA_shapes.pkl)
- [Tree-Cycles](https://github.com/divelab/DIG_storage/blob/main/xgraph/datasets/Tree_Cycles.pkl)
- [Cora](https://github.com/kimiyoung/planetoid/tree/master/data)
- [YelpChi](https://github.com/AI4Risk/antifraud/blob/main/data/YelpChi.zip)
- [Coauthor CS](https://github.com/shchur/gnn-benchmark/raw/master/data/npz/)
- [Amazon](https://docs.google.com/uc?export=download&id=17qhNA8H1IpbkkR-T2BmPQm8QNW5do-aa&confirm=t)

## Structure
- checkpoints: store the trained model.
- config: the parameters of the algorithm and model.
- datasets: datasets used in the experiments.
- dataset.py: datasets processing and loading.
- gnnNets.py: model parameters and architecture.
- SliceGX.py/Slice_MS.py/Slice_MM.py: the SliceGX algorithm.
- train_model.py: train the model.
- utils.py: some help functions.
- Gnnexplainer.py/graphmask.py/pgexplainer_edges.py/subgraphx.py/random_explain.py: baselines.

## Usage
1. Download datasets.
2. Configure the training parameters and run train_gnn.py to train the model(stored in checkpoints):
    > learning_rate: 0.001<br>
    > weight_decay: 5e-4<br>
    > milestones: None<br>
    > gamma: None<br>
    > batch_size: 1<br>
    > num_epochs: 2000<br>
    > num_early_stop: 0<br>
    > gnn_latent_dim:<br>
    >     - 20<br>
    >     - 20<br>
    >     - 20<br>
    > gnn_dropout: 0.0<br>
    > add_self_loop: True<br>
    > gcn_adj_normalization: False<br>
    > gnn_emb_normalization: False<br>
    > graph_classification: False<br>
    > node_classification: True<br>
    > gnn_nonlinear: 'relu'<br>
    > readout: 'identity'<br>
    > fc_latent_dim: [ ]<br>
    > fc_dropout: 0.0<br>
    > fc_nonlinear: 'relu'<br>
    > concate: False<br>
    
![train](Figures/train.png)

4. Config the algorithm parameters in config folder.
   > dataset_root: 'datasets'<br>
   > dataset_name: 'ba_shapes'<br>
   > random_split_flag: False<br>
   > data_split_ratio: [0.8, 0.1, 0.1]<br>
   > seed: 2<br>
   > num_classes: 4<br>
   > K: [2, 4, 6, 8, 10]<br>
   > h: [ 0.1,0.2,0.3 ]<br>
   > theta: [ 0.1,0.2,0.3 ]<br>
   > gamma: 0.5<br>
6. Run Gnnexplainer.py/graphmask.py/pgexplainer_edges.py/subgraphx.py/random_explain.py or SliceGX.py/Slice_MS.py/Slice_MM.py to generate the explanations.

![SliceGX](Figures/SliceGX.png)

## Figures
![result](Figures/result.png)
