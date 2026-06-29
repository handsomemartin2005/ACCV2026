import torch
from ultralytics import RTDETR
from thop import clever_format
if __name__ == '__main__':
    # choose your yaml file
    # ../Graph-MDETR/ultralytics/cfg/models/rt-detr/gcn-rtdetr-mamba.yaml
    model = RTDETR('../Graph-MDETR/ultralytics/cfg/models/rt-detr/gcn-rtdetr-mamba.yaml')
    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[224, 224])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()

    # flops, params = clever_format([34908083, 41989043], "%.3f")  # 计算MACs -> Flops = 2 * MACs
    #
    # print("Total_FLOPs: %s" % (flops))
    # print("Total_params: %s" % (params))