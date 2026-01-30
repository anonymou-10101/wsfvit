"""
Grad-CAM visualization
Support for mixvit, moganet, maxvit, davit, 
Modified from: https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py
 

"""
import torch

import numpy as np
import argparse

from timm.data import create_transform
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt

import models

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='MixViT Grad Cam Visuallization')
parser.add_argument('--data-dir', metavar='DIR', const=None,
                    help='path to dataset')
parser.add_argument('--models', nargs='+',
                    choices=['mixvit', 'resnet50', 'maxvit', 'davit'],
                    default=['mixvit', 'resnet50', 'maxvit', 'davit'],
                    help="type models")
parser.add_argument('--img-size', default=224, type=int,
                    help='size of input image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--checkpoint', metavar='checkpoint_path',
                    help='path to checkpoint')

def build_transform(is_training, input_size=(3, 224, 224), interpolation='bilinear', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), crop_pct=0.9):
    transform = create_transform(
        input_size=input_size,
        is_training=is_training,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
    )
    return transform

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = build_transform(
                is_training=True,
                input_size=(3, 224, 224),
                interpolation='bilinear',
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                crop_pct=0.9,
            )
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        label = torch.tensor(1)
        label = label.unsqueeze(0)
        label = label.to('cuda')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to('cuda')
        return image, label
    
def append_models(args, device):
    
    _models = []
    models_layer = []
    
    if 'maxvit' in args.models:
        maxvit = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=True).to(device).eval()
        _models.append(maxvit)
        models_layer.append(maxvit.stages[3].blocks[-1])
    
    if 'resnet50' in args.models:
        resnet = timm.create_model('resnet50', pretrained=True).to(device).eval()
        _models.append(resnet)   
        models_layer.append(resnet.layer4[-1])

    if 'mixvit' in args.models:
        mixvit = timm.create_model('mixvit_t_224', pretrained=False).to(device).eval()
        state_dict = torch.load(args.checkpoint, weights_only=False)['state_dict']
        mixvit.load_state_dict(state_dict)
        _models.append(mixvit)
        models_layer.append(mixvit.stages[3].blocks[-1])
        
    if 'davit' in args.models:
        davit = timm.create_model('davit_tiny', pretrained=True).to(device).eval()
        _models.append(davit)
        models_layer.append(davit.stages[3].blocks[-1])
        
    return _models, models_layer


def visualize(args):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    data_path = args.data_dir
    data_loader = CustomImageDataset(data_path)
    
    models, models_layer = append_models(args, device)
        
    for batch_idx, (input, _) in enumerate(data_loader):
        
        n_cols = len(models) + 1          
        _, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        axes = np.atleast_1d(axes) 
        
        input = input.to(device)
        
        input_img = input[0].cpu().permute(1, 2, 0).numpy() # [C, H, W] -> [H, W, C]
        input_img = np.clip(input_img, 0, 1)
        
        for model_idx, (model, model_layer) in enumerate(zip(models, models_layer)):
            
            with torch.no_grad():
                logits = model(input)
            class_id = logits.argmax(dim=1).item()         
            targets  = [ClassifierOutputTarget(class_id)]

            with torch.enable_grad(), GradCAM(model=model, target_layers=[model_layer]) as cam:
                gray_scale = cam(input_tensor=input, targets=targets)[0, :]
                
            vis = show_cam_on_image(input_img, gray_scale, use_rgb=True)
            axes[model_idx].imshow(vis)
            axes[model_idx].axis("off")
            axes[model_idx].set_title(f'{model.__class__.__name__}', fontsize=8)

        axes[len(models)].imshow(input_img)
        axes[len(models)].axis("off")
        
        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    
    args = parser.parse_args()
    visualize(args)
