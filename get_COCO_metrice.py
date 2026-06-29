import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets

def parse_opt():
    parser = argparse.ArgumentParser()
    # /home/yhn/Projects/RTDETR-main/dataset/coco_uav/instances_val2017.json
    parser.add_argument('--anno_json', type=str, default='../Graph-MDETR/dataset/coco_uav/instances_val2017.json', help='training model path')
    # /home/yhn/Projects/RTDETR-main/run/val/exp/predictions.json
    parser.add_argument('--pred_json', type=str, default='../Graph-MDETR/run/val/exp/predictions.json', help='data yaml path')
    
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    
    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir='result')