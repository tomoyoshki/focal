# FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space

Authors: **Shengzhong Liu, Tomoyoshi Kimura, Dongxin Liu, Ruijie Wang, Jinyang Li, Suhas Diggavi, Mani Srivastava, Tarek Abdelzaher**

Arxiv Link [[pdf](https://arxiv.org/abs/2310.20071)]

## Overview

This paper proposes a novel contrastive learning framework, called FOCAL, for extracting comprehensive features from multimodal time-series sensing signals through self-supervised training. Existing multimodal contrastive frameworks mostly rely on the shared information between sensory modalities, but do not explicitly consider the exclusive modality information that could be critical to understanding the underlying sensing physics. Besides, contrastive frameworks for time series have not handled the temporal information locality appropriately. FOCAL solves these challenges by making the following contributions: First, given multimodal time series, it encodes each modality into a factorized latent space consisting of shared features and private features that are orthogonal to each other. The shared space emphasizes feature patterns consistent across sensory modalities through a modal-matching objective. In contrast, the private space extracts modality-exclusive information through a transformation-invariant objective. Second, we propose a temporal structural constraint for modality features, such that the average distance between temporally neighboring samples is no larger than that of temporally distant samples. Extensive evaluations are performed on four multimodal sensing datasets with two backbone encoders and two classifiers to demonstrate the superiority of FOCAL. It consistently outperforms the state-of-the-art baselines in downstream tasks with a clear margin, under different ratios of available labels.

## Installation and Requirements

1. **Dependencies**: ```conda create -n [NAME] python=3.10; conda activate [NAME]```

2. **Installation**:

    ```bash
    git clone [repo] [dir]
    cd [dir]
    pip install -r requirements.txt
    ```

## Dataset

### Moving Object Detection (MOD)

MOD is a self-collected dataset using acoustic (8000Hz) and seismic (100Hz) signals to classify moving vehicle types. The pretrain dataset includes lots of 10 classes, and the downstream tasks include vehicle classification, distance classification, and speed classification.

- Vehicle data: [Box link to appear](https://www.box.com)
- Distance speed raw data: [Box link](https://uofi.box.com/s/8yffx3417mrxbdsqqtder4d7kk7rryb8)

### Data preprocessing

1. Update directories in `src/data_preprocess/MOD/proprocessing_configs.py`
2. `cd src/data_preprocess/MOD/`
3. Extract samples from raw data
4. Partition data
5. Please update the index path configurations in `src/data/*.yaml`

#### Sample Extraction

##### Extract pretrain data

```bash
python extract_pretrain_samples.py
```

##### Extract supervised data

```bash
python extract_samples.py
```

##### Extract speed and distance data

```bash
python extract_samples_speed_distance.py
```

#### Data Partition

##### Partition pretrain data

```bash
python partition_data_pretrain.py
```

##### Partition supervised data

```bash
python partition_data.py
```

##### Partition speed_distance data

```bash
python partition_data_speed_distance
```

## Usage

### Training + Fine-tuning

#### Documentation

Please download the dataset from the provided links in the Dataset section. The arguments to run the training program can be found by running

```bash
python train.py --help
```

#### Supervised Training

The following command will run supervised training for DeepSense (`-model=DeepSense`) or Transformer (`-model=SW_Transformer`).

```bash
python train.py -model=[MODEL] -dataset=[DATASET]
```

#### FOCAL Pre-training

```bash
python train.py -model=[MODEL] -dataset=[DATASET] -learn_framework=FOCAL
```

#### FOCAL Fine-tuning with specific downstream task

```bash
python train.py -model=[MODEL] -dataset=[DATASET] -learn_framework=FOCAL -task=[TASK] -stage=finetune -model_weight=[PATH TO MODEL WEIGHT]
```

- Note that the path to the model weight is the folder under the generated `weights/[DATASET]_[MODEL]/` directory.

### Testing

#### Supervised

```bash
python test.py -model=[MODEL] -dataset[DATASET] -model_weight=[PATH TO MODEL WEIGHT]
```

#### FOCAL

```bash
python test.py -model=[MODEL] -dataset[DATASET] -stage=finetune -task=[TASK] -model_weight=[PATH TO MODEL WEIGHT]
```

## Model Configs

See the file `src/data/*.yaml` for specific model configurations for each dataset.

## License

This project is released under the MIT license. See `LICENSE` for details.

## Citation

```latex
@inproceedings{liu2023focal,
  title = {FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space},
  author = {Liu, Shengzhong and Kimura, Tomoyoshi and Liu, Dongxin and Wang, Ruijie and Li, Jinyang and Diggavi, Suhas and Srivastava, Mani and Abdelzaher, Tarek},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023}
}
```

## Contact

For any issue regarding the paper or the code, please contact us.

- Shengzhong Liu: <shengzhong@sjtu.edu.cn>
- Tomoyoshi Kimura: <tkimura4@illinois.edu>
