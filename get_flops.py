"""
please install requirements before running
`pip install -r requirements.txt`


"""
import argparse 

import torch
from timm.models import create_model
from fvcore.nn import FlopCountAnalysis, flop_count_table

import models

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mixvit_tiny_224', type=str,
                    help='model architecture (default : mixvit_tiny_224)')
parser.add_argument('--img-size', default=224, type=int,
                    help='input image size (default : 224)')

def get_flops(args):

    model = create_model(args.model)
    model.eval()
    
    flops = FlopCountAnalysis(model, torch.randn((1, 3, args.img_size, args.img_size)))
    
    print(flop_count_table(flops, max_depth=4))
    
    params = sum([m.numel() for m in model.parameters()])/1e6
    flops_count = flops.total() / 1e9
    print(f'PARAMS (M) :{ params: .3f} | FLOPS (G) : {flops_count :.3f}')

if __name__ == "__main__":
    args = parser.parse_args()
    get_flops(args)
