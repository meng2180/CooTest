# CooTest

This is the official implementation of paper. "[CooTest: An Automated Testing Approach for V2X Communication Systems](https://arxiv.org/abs/2408.16470)". 

## Installation

All experiments are conducted on a server with an Intel i7-10700K CPU(3.80 GHz), 32 GB RAM, and an NVIDIA GeForce RTX 4070 GPU (12GB VRAM).

### Init Dataset

You need to check the [V2V4real](https://mobility-lab.seas.ucla.edu/v2v4real/) website and download the test datasets test1, test2, test3. 

### Basic Dependency

To set up the codebase environment, do the following steps:

Create conda environment (python >= 3.8)

```shell
conda create -n coo-test python=3.9
conda activate coo-test
```

Pytorch Installation (>= 1.12.0 Required)

```shell
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

spconv 2.x Installation

```shell
pip install spconv-cu113
```

Install other dependencies

```shell
pip install -r requirements.txt
python setup.py develop
```

Install bbx nms calculation cuda version

```shell
python opencood/utils/setup.py build_ext --inplace
```

### Model Donwload

You need to download the cooperative 3D detection models [here](https://github.com/ucla-mobility/V2V4Real?tag=readme-ov-file#benchmark), and unzip them in the model folder.

```shell
model
├── attfuse
├── early_fusion
├── late_fusion
├── PointPillar_Fcooper
├── PointPillar_V2VNet
└── PointPillar_V2XViT
```



## Experiments

### RQ1

You need to select different models and calculate the AP values for different transformations (including unchanged) results.

```shell
python rq_tools/rq1_eval.py --dataset_dir ${dataset}/rq1 --model_dir ${model}/late_fusion
```

Arguments Explanation:

- `model_dir`: the path to your saved model, e.g. `model/early`, meaning you want to use "early_fusion "model for test. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files. 
- `dataset_dir`: the "test" dataset path.

The output content includes the results of applying a single operator and the mixed  operator to the original dataset "test".

### RQ2

#### This section includes the following steps:

1. Split the original dataset for training, inference, and testing respectively;
2. Select and save data for retraining using V2X-oriented guided transformation;
3. Generate augmented data and test the erroneous behaviors of cooperative perception tasks using MRs.

```shell
python rq_tools/rq2.py --dataset_dir ${dataset}/test --model_dir ${model}/${model_name}
```

The selected data is saved in the `${dataset}/rq2/rq2_select` directory.

### RQ3

#### 1. Retrain  the models

```shell
python rq_tools/rq3_train.py --dataset_dir ${dataset}/rq2/rq2_select --model_dir ${model}/${model_name}
```

#### 2. Test the effect of retrain models

```shell
python rq_tools/rq3_eval.py --dataset_dir ${dataset}/rq2/test --model_dir ${model}/retrained/${model_name}
```

## Dataset structure

After generate the transformation datasets, the recommended dataset format like this:

```shell
├── coo-test
│   ├── test
│   ├── rq1
│      ├── rq1_1
│      ├── rq1_2
│   ├── rq2
│      ├── test 
│      ├── train
│      ├── rq2_select
│      ├── rq2_det_box
│   ├── rq3
│      ├── rq3_test
└──    ├── rq3_det_box
```



## Citation

```shell
@inproceedings{guo2024cootest,
  title={CooTest: An Automated Testing Approach for V2X Communication Systems},
  author={Guo, An and Gao, Xinyu and Chen, Zhenyu and Xiao, Yuan and Liu, Jiakai and Ge, Xiuting and Sun, Weisong and Fang, Chunrong},
  booktitle={Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis},
  pages={1453--1465},
  year={2024}
}
```



























