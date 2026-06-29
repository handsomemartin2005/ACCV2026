import contextlib
import io

import torch

from ultralytics import RTDETR


def smoke(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        model = RTDETR(cfg)
    net = model.model.eval().to('cpu')
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        y = net(x)
    pred = y[0] if isinstance(y, tuple) else y
    print(f'{cfg}: pred={tuple(pred.shape)}, layers={len(net.model)}')
    if 'sch-' in cfg:
        backbone = net.model[0]
        print(f'  use_real_clip={backbone.semantic_prior.use_real_clip}')
        print(f'  route_scores={tuple(backbone.last_route_scores.shape)}')
        print(f'  route_indices={backbone.last_route_indices.tolist()}')
        print(f'  semantic_response={tuple(backbone.last_semantic_response.shape)}')


if __name__ == '__main__':
    smoke('ultralytics/cfg/models/rt-detr/gcn-rtdetr-mamba.yaml')
    smoke('ultralytics/cfg/models/rt-detr/sch-rtdetr-mamba.yaml')
