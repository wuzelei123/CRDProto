# Few-Shot Learning with Noisy Labels

## 🔧 Environment Setup

All dependencies are listed in `environment.yml`. The experiments in the paper were conducted with **Python 3.9.6** and **PyTorch 1.9.0**. Other versions may work but are untested.

Create the conda environment using:

```bash
conda env create --file environment.yml
Models use a two-stage approach: first pre-train the backbone, then freeze it before training few-shot classifiers.

To train the Conv4 backbone on MiniImageNet:

```bash
bash scripts/train_backbone.sh
This will train the backbone and evaluate ProtoNet at various noise levels.

This project is based on and modifies the official implementation of:

> Kevin J. Liang, Samrudhdhi B. Rangrej, Vladan Petrovic, and Tal Hassner.  
> **Few-shot Learning with Noisy Labels**.  
> In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 9089–9098.  
> [Paper (CVPR Open Access)](https://openaccess.thecvf.com/content/CVPR2022/html/Liang_Few-Shot_Learning_With_Noisy_Labels_CVPR_2022_paper.html)

BibTeX:
```bibtex
@inproceedings{liang2022few,
  title={Few-shot learning with noisy labels},
  author={Liang, Kevin J and Rangrej, Samrudhdhi B and Petrovic, Vladan and Hassner, Tal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9089--9098},
  year={2022}
}
