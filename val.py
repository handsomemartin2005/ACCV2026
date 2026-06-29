import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import pandas as pd

if __name__ == '__main__':
    model = RTDETR('../Graph-MDETR/run/train/exp/weights/best.pt')
    metrics = model.val(data='../Graph-MDETR/dataset/coco_UAV.yaml',
              split='val',
              imgsz=640,
              batch=16,
              save_json=True, # if you need to cal coco metrice
              project='run/val',
              name='exp',
              )

    # # 假设metrics是评估结果对象
    # # 提取Precision和Recall数据
    # recalls = metrics.curves_results[0][0]  # (1000) 包含所有类别的Recall
    # precisions = metrics.curves_results[0][1]  # (17, 1000) 包含所有类别的Precision
    #
    # csv_recalls = recalls
    # csv_precisions = precisions
    #
    # # 创建一个DataFrame
    # csv_recalls = pd.DataFrame(csv_recalls)
    # csv_precisions = pd.DataFrame(csv_precisions)
    #
    # # 将DataFrame保存为CSV文件
    # csv_recalls.to_csv('/home/robot/桌面/csv_recalls.csv', index=False)
    # csv_precisions.to_csv('/home/robot/桌面/csv_precisions.csv', index=False)

    print('end')

