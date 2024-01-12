## A Novel Deep Video Watermarking Framework with Enhanced Robustness to H.264/AVC Compression

### ACM MM 2023
**Yulin Zhang, Jiangqun Ni, Wenkang Su, and Xin Liao**

## Introduction
This repository is a code release for the paper found [here](https://doi.org/10.1145/3581783.3612270). The paper focus on deep video watermarking with temporal robustness and invisibility. The main contributions are the proposed temporal-associated feature extraction block (TAsBlock),  differentiable video compression simulator(DiffH264), and spatial/temporal mask loss.

## Citation
If you find our work useful, please consider citing:
```
@inproceedings{zhang2023novel,
  title={A Novel Deep Video Watermarking Framework with Enhanced Robustness to H. 264/AVC Compression},
  author={Zhang, Yulin and Ni, Jiangqun and Su, Wenkang and Liao, Xin},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={8095--8104},
  year={2023}
}
```

## License
The models are free for non-commercial and scientific research purpose. Please mail us for further licensing terms.

## References
1. The optic flow estimation code is based on [sniklaus/pytorch-spynet](https://github.com/sniklaus/pytorch-spynet). The original paper is 
```
@inproceedings{Ranjan_CVPR_2017,
    author = {Ranjan, Anurag and Black, Michael J.},
    title = {Optical Flow Estimation Using a Spatial Pyramid Network},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2017}
}

```
2. The code for intra compression and residual compression is based on [mlomnitz/DiffJPEG](https://github.com/mlomnitz/DiffJPEG).
