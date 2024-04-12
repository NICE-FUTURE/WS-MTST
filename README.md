# WS-MTST
Official implementation of "WS-MTST: Weakly Supervised Multi-Label Brain Tumor Segmentation With Transformers", <a href="https://ieeexplore.ieee.org/document/10269670" target="_blank">link</a>.

## Quick Start
```
pip install -r requirements.txt
bash run.sh
```

## Data preparation

Please refer to <a href="https://github.com/Merofine/BrainTumorSegmentation/blob/main/data/BraTS2Dpreprocessing-master/GetTestingSetsFrom2019.ipynb" target="_blank">this code</a>.

## Acknowledgements
Our code is based on <a href="https://github.com/NVlabs/SegFormer" target="_blank">SegFormer</a> and <a href="https://github.com/lightly-ai/lightly" target="_blank">lightly</a>.

## Cite
```
@ARTICLE{10269670,
    author={Chen, Huazhen and An, Jianpeng and Jiang, Bochang and Xia, Lili and Bai, Yunhao and Gao, Zhongke},
    journal={IEEE Journal of Biomedical and Health Informatics}, 
    title={WS-MTST: Weakly Supervised Multi-Label Brain Tumor Segmentation With Transformers}, 
    year={2023},
    volume={27},
    number={12},
    pages={5914-5925},
    doi={10.1109/JBHI.2023.3321602}
}
```
