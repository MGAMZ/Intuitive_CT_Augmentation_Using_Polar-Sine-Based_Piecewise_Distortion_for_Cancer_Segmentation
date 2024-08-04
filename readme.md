# Intuitive CT Augmentation Using Polar-Sine-Based Piecewise Distortion for Cancer Segmentation

*Preview* Implementation of the paper

## Authors

- Yiqin Zhang *(First Author)*
- Qingkui Chen *(Corresponding Author)*
- Chen Huang
- Zhengjie Zhang
- Meiling Chen
- Zhibing Fu

*The First Author's Email:* [zyqmgam@163.com](mailto:zyqmgam@163.com)

*Corresponding Author's Email:* [chenqingkui@usst.edu.cn](mailto:chenqingkui@usst.edu.cn)

**Affiliations:**

- School of Optical-electrical and Computer Engineering, University of Shanghai for Science and Technology, Shanghai, 200093, China
  - Yiqin Zhang
  - Qingkui Chen
  - Meiling Chen
  - Zhibing Fu

- School of Health Science and Engineering, University of Shanghai for Science and Technology, Shanghai, 200093, China
  - Zhengjie Zhang

- Department of Gastrointestinal Surgery, Shanghai General Hospital, Shanghai Jiao Tong University, Shanghai, 201600, China
  - Chen Huang

## Graphical Abstact

![GraphicalAbstract](./PaperWriting/5.LatexRefined/TexProject/Figures/Graphical Abstract.png)

## Abstract

Most data-driven models for CT image analysis rely on universal augmentations to improve performance. Experimental evidence has confirmed their effectiveness, but the unclear mechanism underlying them poses a barrier to the widespread acceptance and trust in such methods within the medical community. We revisit and acknowledge the unique characteristics of CT images apart from traditional digital images, and consequently, proposed a CT-specific augmentation algorithm that is more elastic and aligns well with CT scan procedure. The method performs piecewise affine with sinusoidal distorted ray according to radius on polar coordinates, thus simulating uncertain postures of human lying flat on CT scan table. Our method could generate human visceral distribution without affecting the fundamental relative position on axial plane. Two non-adaptive algorithms, namely Meta-based Scan Table Removal and Similarity-Guided Parameter Search, are introduced to bolster robustness of our augmentation method. Experiments show our method improves accuracy across multiple famous segmentation frameworks without requiring more data samples. Our preview code is available in: https://github.com/MGAMZ/PSBPD.

## Prerequisites

Please see `./requirement/*`.

- Ubuntu 22.02 on WSL2
- Ampere or above GPU
- MMengine 0.10.4
- MMsegmentation 1.2.2
- PyTorch 2.5.0+ (current nightly)
- CuDNN 9.1.0+
- [Our Medical Image Analysis Toolkit](https://github.com/MGAMZ/mgam_datatoolkit)

## For Experiments' Details

All experiments configurations are available in `./configs` and based on mmengine framework. So you may be familar to openmim project to better reproducibility. OpenMIM docs are available in [mmenging](https://github.com/open-mmlab/mmengine) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

I (The first author) am a contributor to the openmim open source project. If you have any questions, please feel free to reach out to me via email: `312065559@qq.com`.

## For Usage of The Proposed Method

Please access our [Medical Image Analysis Toolkit](https://github.com/MGAMZ/mgam_datatoolkit), where our previous work and this work are included.

# Reference

To be updated.
