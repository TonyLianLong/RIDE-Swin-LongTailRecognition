# RIDE: Long-tailed Recognition by Routing Diverse Distribution-Aware Experts (Swin-Transformer implementation)

by [Xudong Wang](http://people.eecs.berkeley.edu/~xdwang/), [Long Lian](https://github.com/TonyLianLong/), [Zhongqi Miao](https://scholar.google.com/citations?user=at4m2mYAAAAJ&hl=en), [Ziwei Liu](https://liuziwei7.github.io/) and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/) at UC Berkeley, ICSI and NTU

<em>International Conference on Learning Representations (ICLR), 2021. **Spotlight Presentation**</em>

[Project Page](http://people.eecs.berkeley.edu/~xdwang/projects/RIDE/) | [PDF](http://people.eecs.berkeley.edu/~xdwang/papers/ICLR2021_RIDE.pdf) | 
[Preprint](https://arxiv.org/abs/2010.01809) | [OpenReview](https://openreview.net/forum?id=D9I3drBz4UC) | [Slides](http://people.eecs.berkeley.edu/~xdwang/projects/RIDE/ICLR2021-RIDE-10mins-V4.pdf ) | [Citation](#citation)

<img src="https://github.com/frank-xwang/RIDE-LongTailRecognition/raw/main/title-img.png" width="100%" />

This repository contains an official re-implementation of [RIDE](https://github.com/frank-xwang/RIDE-LongTailRecognition) on [**Swin-Transformer**](https://github.com/microsoft/Swin-Transformer) from the authors. Here is [RIDE with ResNet and ResNeXt implementation](https://github.com/frank-xwang/RIDE-LongTailRecognition). It has also experimental support for LDAM-DRW. Further information regarding the implementation, please contact [Long Lian](mailto:longlian@berkeley.edu). Information regarding RIDE, please contact [Xudong Wang](mailto:xdwang@eecs.berkeley.edu).

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{wang2020long,
  title={Long-tailed Recognition by Routing Diverse Distribution-Aware Experts},
  author={Wang, Xudong and Lian, Long and Miao, Zhongqi and Liu, Ziwei and Yu, Stella},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

# Dataset preparation
We only suppoert ImageNet-LT now. We use the same dataset format as RIDE. Refer to [here](https://github.com/frank-xwang/RIDE-LongTailRecognition#dataset-preparation) to download and prepare the dataset in `dataset` directory.

# Training
The following code is adapted for 8x 2080 Ti. You can change the arguments if you use other hardware.
## Swin-Transformer Tiny
### 2 Experts
```
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/swin_tiny_patch4_window7_224_lt_2_experts_ride.yaml --batch-size 64 --accumulation-steps 2
```
### 3 Experts
```
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/swin_tiny_patch4_window7_224_lt_3_experts_ride.yaml --batch-size 32 --accumulation-steps 4
```
## Swin-Transformer Small
### 2 Experts
```
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/swin_small_patch4_window7_224_lt_2_experts_ride.yaml --batch-size 32 --accumulation-steps 4
```
### 3 Experts
```
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/swin_small_patch4_window7_224_lt_3_experts_ride.yaml --batch-size 128 --accumulation-steps 1 --use-checkpoint
```

# Evaluation
```
OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg [Your config file] --batch-size 32 --accumulation-steps 4 --resume [Your checkpoint file] --eval
```

# Model Zoo

The models here are the ones *without* expert assignment module. More about expert assignment can be found in the note below. 

| Model                | #Experts  | Overall Accuracy | Many Accuracy | Medium Accuracy | Few Accuracy | Download |
| -------------------- | --------- | ---------------- | ------------- | --------------- | ------------ | -------- |
| **RIDE (Tiny)**      | 2         |  53.6            | 65.0          | 50.0            | 34.2         | [Link](https://drive.google.com/drive/folders/1rt9FKQp-mNue-hfxFt7MR4U_QTrvn10j?usp=sharing)     |
| **RIDE (Tiny)**      | 3         |  54.4            | 65.7          | 50.7            | 35.5         | [Link](https://drive.google.com/drive/folders/1nFjdnc7r-XWlYEBE4rkV1yJMYg3IBoEG?usp=sharing)     |
| **RIDE (Small)**     | 2         |  56.3            | 67.4          | 52.9            | 37.0         | [Link](https://drive.google.com/drive/folders/18PXo0oiDwz8dQ3dYrzQnUinJ42AeYcsO?usp=sharing)     |
| **RIDE (Small)**     | 3         |  56.2            | 67.0          | 53.0            | 37.5         | [Link](https://drive.google.com/drive/folders/1j9QwP-_OFwzQaPj2Vp5YW-dWXd5aVcJG?usp=sharing)     |

# Notes
This implementation is still experimental. If you have any questions regarding this implementation, please contact us at `longlian at berkeley.edu` and `xdwang at eecs.berkeley.edu`. If you have any RIDE questions, please refer to [FAQ](https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/FAQ.md) in the RIDE repo.

Expert assignment is implemented in our own implementation and the Swin-Transformer results reported in the updated version of the arXiv paper are with expert assignment. However, in this implementation, we use a different way of implementation when compared to our ResNet version, which make the code much simpler to understand but expert assignment harder to implement. We decide to release the implementation without expert assignment to make easier understanding about RIDE. If you want to implement expert assignment on your own, here are several places to change:

1. You need to change Swin-Transformer to allow dynamic control on what expert to use.
2. You need to add the expert assignment module (as in our [EAResNet](https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/model/fb_resnets/EAResNet.py)). The module itself can be identical.
3. You need to add another training option that only allows training the expert assignment module. We found that using SGD rather than the AdamW optimizer helps training the expert assignment module (Swin-Transformer on its own still uses AdamW).

Feel free to reach out to us for more details.
