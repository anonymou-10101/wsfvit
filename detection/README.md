# Applying WSF-ViT to Object Detection
This repository provides a PyTorch implementation of WSF-ViT for object detection and instance segmentation on the COCO dataset, using Mask R-CNN and Cascade Mask R-CNN. The implementation is based on MMDetection.

### Envirnoment Setup
- **OS**  : Linux (tested)  MMDetection itself also supports Windows / macOS  
- **Python** : 3.9  
- **CUDA**  : 11.3  
- **PyTorch**: 1.12.1 

You can test this project either with Docker or through a local installation.

### Docker Installation (recommended)
Clone this repository from GitHub and build the Docker image using the provided Dockerfile.
```shell
git clone https://github.com/anonymou-10101/wsfvit.git WSF-ViT && cd ./WSF_ViT/detection/Docker && docker build -t wsfvit:det .
```

After building the Docker image, you can run the project. We highly recommend testing it on a GPU.
```shell
docker run -it --gpus all -v /path/to/you/WSF-ViT:/app/ wsfvit:det
```

### Local Installation

__Step 0.__ Download and install Miniconda from the official website.

__Step 1.__ Create a conda environment and activate it.
```shell
conda create --name openmmlab python=3.9 -y
conda activate openmmlab
```

__step 2.__ Install PyTorch following official instructions, e.g.

On GPU platforms:
```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

__step 3.__ Install MMDetection and MMCV
```shell
conda install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```
```shell
conda install mmdet
````

__step 4.__ Install Dependenceis
```shell
pip install -r requirements.txt
```




## Result and models on COCO

### WSF-ViT + Mask R-CNN
|   Method   | Backbone  |   Pretrain  | Resolution | Params | FLOPS  | Lr schd | box mAP | AP50 | AP75 | mask mAP | AP50 | AP75 | Download |
| :--------: | :-------: | :---------: | :--------: | :----: | :----: | :-----: | :-----: | :--: | :--: | :------: | :--: | :--: | :------: |
| Mask R-CNN | WSF-ViT-T | ImageNet-1K | 1120 x 896 |  42.7M | 254.6G |  MS 3x  |   48.6  | 70.6 | 53.6 |   43.7   | 67.6 | 47.2 | [here](https://drive.google.com/file/d/1h0E4pVdz3QOiT_5eg46FPd5kc7Yr3NBt/view?usp=drive_link) |



### WSF-ViT + Cascade Mask R-CNN
|       Method       | Backbone  |   Pretrain  | Resolution | Params | FLOPS |  Lr schd   | box mAP | AP50 | AP75 | mask mAP | AP50 | AP75 | Download |
| :----------------: | :-------: | :---------: | :--------: | :----: | :---: | :--------: | :-----: | :--: | :--: | :------: | :--: | :--: | :------: |
| Cascade Mask R-CNN | WSF-ViT-T | ImageNet-1K | 1120 x 896 |  80.5M |  733G | GIOU+MS 3x |  52.3   | 71.1 | 56.6 |  45.2    | 68.4 | 49.0 | [here](https://drive.google.com/file/d/1XS2FZre0QcdbC4teWoDZBt9IDOvBjJfu/view?usp=drive_link) |
| Cascade Mask R-CNN | WSF-ViT-S | ImageNet-1K | 1120 x 896 |  98.9M |  788G | GIOU+MS 3x |  53.2   | 72.2 | 57.9 |  46.0    | 69.6 | 49.9 | [here](https://drive.google.com/file/d/1QvN5exdoPUH-aseNuf2oBudX5RbbKWrs/view?usp=drive_link) |
| Cascade Mask R-CNN | WSF-ViT-M | ImageNet-1K | 1120 x 896 | 126.7M |  885G | GIOU+MS 3x |  53.6   | 72.4 | 58.2 |  46.4    | 69.8 | 50.5 | [here](https://drive.google.com/file/d/1521Uu4TtgDfMaVdH2cIEtya_8L0Xu0YS/view?usp=drive_link) |



### calc flops and params
```shell
python ./tools/analysis_tools/get_flops.py ./configs/wsfvit/mask_rcnn_wsfvit_t_3x_coco.py --shape 1120 896
```

## Training
Start training with the config as :
```shell
bash ./tools/dist_train.sh ./configs/wsfvit/mask_rcnn_wsfvit_t_3x_coco.py num_of_gpus
```

## Evaluation
To evalute the trained model, run:
```shell
bash ./tools/dist_test.sh ./configs/wsfvit/mask_rcnn_wsfvit_t_3x_coco.py /path/to/checkpoint num_of_gpus --out results.pkl --eval bbox segm
```

## Acknowledgment
Our implemnation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.
- [MMDetection](https://github.com/open-mmlab/mmdetection/tree/2.x)
- [MogaNet](https://github.com/Westlake-AI/MogaNet)
- [SwinTransformer](https://github.com/microsoft/Swin-Transformer)

