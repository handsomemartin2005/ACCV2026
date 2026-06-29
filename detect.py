import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('../Graph-MDETR/run/train/exp2/weights/best.pt')  # select your model.pt path
    model.predict(source='../Graph-MDETR/dataset/coco_net12/images/val',  # '/home/yhn/Projects/RTDETR-main/dataset/coco_net12/images/val
                  project='run/detect',
                  name='exp1',
                  save=True,
                  line_width=12,  # 8 10 12 14
                  conf=0.45,
                  visualize=False  # visualize model features maps
                  )