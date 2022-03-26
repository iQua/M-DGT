# cvpr22
This is the source code of our paper for CVPR 2022. The original idea of this paper is the multi-modal dynamic graph transformer framework for progressive learning on visual grounding.


![LICENSE](https://img.shields.io/badge/license-Apache-green?style=flat-square)
![Python](https://img.shields.io/badge/python-3.5%20%7C%203.7-blue.svg?style=flat-square)
![PyTorch](https://img.shields.io/badge/pytorch-1.5.0-%237732a8?style=flat-square)


#### Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Datasets](#datasets)
* [Structure](#structure)
* [Operation](#operation)
* [Performance](#performance)
* [Acknowledgements](#acknowledgements)


## Introduction

The multi-modal dynamic graph transformer (M-DGT) framework is built upon a novel idea that models the learning of the visual grounding as the graph transformation. To accomplish this, there are two crucial components, including the multi-modal node transformer and the graph transformer. With the anchor in the image as the node, the multi-modal node transformer firstly constructs a graph based on the spatial positions of anchors. Then it continuously produces the 2D transformation coefficients to modify these spatial positions to approach the ground truth regions. Then, during this process, the graph transformer optimizes the structure of the graph by removing nodes and unnecessary edges to motivate efficient learning. Therefore, the whole framework can be regarded as generating a series of dynamic graphs that gradually shrink to the target regions. The performance of the M-DGT is measured in two parts, including visual grounding and phrase grounding. The ReferItGame and RefCOCO datasets are used to measure the performance of the M-DGT on visual grounding, while the Flickr30K Entities dataset is utilized to test the M-DGT on phrase grounding.


## Getting Started

Please install the necessary Python packages before running the code. The main Python packages are PyTorch, torchvision, mmcv, opencv, albumentations.

Then, you need to download three corresponding datasets and then set the path in the 'common_flags.py'.


## Datasets

This paper utilizes the following three classic datasets:

1. ReferItGame dataset. Please check the data provider souce code for this dataset in the 'datasets/ReferItGame_provider'.
2. Flickr30K Entities dataset. Please check the data provider souce code for this dataset in the 'datasets/F30KE_provider'.
3. RefCOCO dataset. Please check the data provider souce code for this dataset in the 'datasets/ReferItGame_provider'.


## Structure

> Folder structure and functions for M-DGT

    .
    ├── datasets                        # Three datasets used in experiments
    ├── experiments                     # The directory used to save the model and logging file
    ├── learning                        # The train, evaluation, and losses parts
    ├── models                          # The components and main structures of M-DGT
    ├── preprocess                      # The function used to preprocess the original image 
    ├── visualization                   # Visualization part of M-DGT
    └── common_flags.md                 # Set common paths for datasets
    └── f30k_eval.py                    # Evaluate the model on eval/test sets of the Flickr30K Entities dataset.
    └── f30k_train.py                   # Train the model on the train set of the Flickr30K Entities dataset.
    └── refcoco_eval.py                 # Evaluate the model on eval/tests set of ReferItGame/RefCOCO datasets.
    └── refcoco_train.py                # Train the model on the train set of ReferItGame/RefCOCO datasets.
    └── README.md


## Operation

After setting the environment and three datasets, the model can be trained directly by running:

1. Operations on the Flickr30K Entities dataset
>  Train the model
```console
cvpr@cvpr22:~$ python f30k_train.py
```
>  Test the model; Set 'phase' to switch between test and val.
```console
cvpr@cvpr22:~$ python f30k_eval.py
```

2. Operations on the RefCOCO/ReferItGame dataset
>  Train the model; Set 'data_name' and 'split_type' to switch between different datasets.
```console
cvpr@cvpr22:~$ python refcoco_train.py
```
>  Test the model; Set 'phase' to switch between test and val.
```console
cvpr@cvpr22:~$ python refcoco_eval.py
```


## Performance

### Flickr30K Entities
| (%)           | top-1 accuracy|
| ------------- |:-------------:|
| SOTA          | 76.74 [Learning Cross-Modal Context Graph ](https://arxiv.org/abs/1911.09042) |
| M-DGT         | 79.97         | 


### RefCOCO
| (%)           | ReferCOCO    | ReferCOCO+ | ReferCOCOg
| ------------- |:-------------:|:-------------:|:-------------:|
| type          | Val / TestA / TestB   | Val / TestA / TestB |Val / Test |
| SOTA          | 82 / 81.20 / 84.00   | 66.6 / 67.6 / 65.5 | 75.73 / 75.31 |
| MMDGT         | 85.37 / 84.82 / 87.11  | 70.02 / 72.26 / 68.92 | 79.21 / 79.06
    


## Acknowledgements

Our implementations refer to the source code from the following repositories and users:

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)

* [Dynamic Graph CNN](https://github.com/WangYueFt/dgcnn)

* [Graph Transformer Networks](https://github.com/seongjunyun/Graph_Transformer_Networks)

* [Transformer](https://github.com/jayparks/transformer)

* [PositionalEncoding2D](https://github.com/wzlxjtu/PositionalEncoding2D)

* [refer](https://github.com/lichengunc/refer)
