import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import torch

from ultralytics import RTDETR

if __name__ == '__main__':
    root = Path(__file__).resolve().parent

    model = RTDETR(root / 'ultralytics/cfg/models/rt-detr/sch-rtdetr-mamba.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=root / 'dataset/coco_UAV.yaml',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=16,
                device='0' if torch.cuda.is_available() else 'cpu',
                # resume='/home/robot/Projects/RTDETR-main/run/train/exp/weights/last.pt', # last.pt path
                name='sch_exp',
                workers=4,
                save_period=1)
