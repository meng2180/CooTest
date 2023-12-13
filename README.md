# CooTest

## Test dataset dawnload

[dataset](https://mobility-lab.seas.ucla.edu/v2v4real/)

## Devkit setup

To set up the codebase environment, do the following steps:

#### 1. Create conda environment (python >= 3.7)

```shell
conda create -n v2v4real python=3.7
conda activate v2v4real
```

#### 2. Pytorch Installation (>= 1.12.0 Required)

Take pytorch 1.12.0 as an example:

```shell
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

#### 3. spconv 2.x Installation

```shell
pip install spconv-cu113
```

#### 4. Install other dependencies

```shell
pip install -r requirements.txt
python setup.py develop
```

#### 5.Install bbx nms calculation cuda version

```shell
python opencood/utils/setup.py build_ext --inplace
```

## RQ1

#### Generate augment datasets

You need to change the configuration file(e.g. `PointPillar_Fcooper/config.yaml `and `nofusion/config.yaml`) ：

```yaml
validate_dir: '/data/v2vreal/rq1/test'
```

```shell
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --data_augment 285 --rq_command rq1
```

Arguments Explanation:

- `model_dir`: the path to your saved model, e.g. `model/PointPillar_Fcooper`, meaning you want to use FCooper model for test. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files. 
- `fusion_method`: indicate the fusion strategy, currently support 'nofusion', 'early', 'late', and 'intermediate'.
- `data_augment`: generates the same size of transformeddatasets as test dataset for each transformation operator

#### RQ1 Test

You need to change the configuration file：

```yaml
validate_dir: '/data/v2vreal/rq1/t1'
```

```shell
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --rq_command rq1
```

## RQ2 and RQ3

#### Generate augment dataset

You need to change the configuration file：

```yaml
validate_dir: '/data/v2vreal/test'
```

```shell
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --data_augment 996 --rq_command rq3
```

#### RQ2: Using a guided method and a random approach, 10% and 15% of the data were selected for retraining

You need to change the configuration file：

```yaml
validate_dir: '/data/v2vreal/Retrain/retrain/augment_data'
```

```shell
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --data_select formula --rq_command rq3
```

#### Retrain

You need to change the configuration file：

```yaml
root_dir: '/data/v2vreal/Retrain/t1'
validate_dit: '/data/v2vreal/test'
```

```shell
python opencood/tools/train_da.py --hypes_yaml hypes_yaml/domain_adaptions/xxx.yaml --model_dir  ${CHECKPOINT_FOLDER}
```

#### RQ3: Test the effect of retrain

You need to change the configuration file：

```yaml
validate_dir: '/data/v2vreal/Retrain/augment_data'
```

```shell
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} --rq_command rq3
```

If you want to go straight to the retrained model, use the model with the `_retrain` suffix.

## File structure

After generat the augment datasets, please put the data in the following structure:

```shell
├── v2v4real
│   ├── t1
│   ├── test
│   ├── rq1
│      ├── test
│      ├── t1
│   ├── Retrain
│      ├── t1
│      ├── augment_data
│      ├── retrain
│      		├── augment_data
```





























