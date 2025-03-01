# GS-Hider: Hiding Messages into 3D Gaussian Splatting
[Xuanyu Zhang](https://xuanyuzhang21.github.io/), [Jiarui Meng](), [Runyi Li](https://villa.jianzhang.tech/people/runyi-li-%E6%9D%8E%E6%B6%A6%E4%B8%80/), [Zhipei Xu](), [Yongbing Zhang](), [Jian Zhang](https://jianzhang.tech/)

[![arXiv](https://img.shields.io/badge/arXiv-<Paper>-<COLOR>.svg)](https://arxiv.org/abs/2405.15118)
[![Home Page](https://img.shields.io/badge/Project_Page-<Website>-blue.svg)](https://xuanyuzhang21.github.io/project/gshider/)

# Abstract
3D Gaussian Splatting (3DGS) has already become the emerging research focus in the fields of 3D scene reconstruction and novel view synthesis. Given that training a 3DGS requires a significant amount of time and computational cost, it is crucial to protect the copyright, integrity, and privacy of such 3D assets. Steganography, as a crucial technique for encrypted transmission and copyright protection, has been extensively studied. However, it still lacks profound exploration targeted at 3DGS. Unlike its predecessor NeRF, 3DGS possesses two distinct features: 1) explicit 3D representation; and 2) real-time rendering speeds. These characteristics result in the 3DGS point cloud files being public and transparent, with each Gaussian point having a clear physical significance. Therefore, ensuring the security and fidelity of the original 3D scene while embedding information into the 3DGS point cloud files is an extremely challenging task. To solve the above-mentioned issue, we first propose a steganography framework for 3DGS, dubbed GS-Hider, which can embed 3D scenes and images into original GS point clouds in an invisible manner and accurately extract the hidden messages. Specifically, we design a coupled secured feature attribute to replace the original 3DGS's spherical harmonics coefficients and then use a scene decoder and a message decoder to disentangle the original RGB scene and the hidden message. Extensive experiments demonstrated that the proposed GS-Hider can effectively conceal multimodal messages without compromising rendering quality and possesses exceptional security, robustness, capacity, and flexibility.

# Install with our feature rasterizer

'''
pip install submodule/diff-gaussian-rasterization
pip install submodule/simple-knn
'''

# Train

```
bash script.sh
```
