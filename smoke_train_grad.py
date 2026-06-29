import argparse
import contextlib
import io

import torch

from ultralytics import RTDETR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        default='ultralytics/cfg/models/rt-detr/sch-rtdetr-mamba.yaml',
        help='RT-DETR YAML config to test.')
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        model = RTDETR(args.cfg)
    net = model.model.to(args.device).train()
    img = torch.rand(1, 3, args.imgsz, args.imgsz, device=args.device)
    batch = {
        'img': img,
        'batch_idx': torch.tensor([0, 0], dtype=torch.long, device=args.device),
        'cls': torch.tensor([[0.], [4.]], dtype=torch.float32, device=args.device),
        'bboxes': torch.tensor(
            [[0.50, 0.50, 0.20, 0.20], [0.30, 0.35, 0.12, 0.16]],
            dtype=torch.float32,
            device=args.device),
    }
    loss, items = net.loss(batch)
    loss.backward()

    print('cfg', args.cfg)
    print('loss', float(loss.detach()))
    print('items', [float(x) for x in items])

    backbone = net.model[0]
    if hasattr(backbone, 'router'):
        router_grad = backbone.router.router[0].weight.grad
        print('router_grad_norm', float(router_grad.norm()))
        print('scale_prior_grad_norm', float(backbone.router.scale_embed.grad.norm()))
        print('structure_prior_grad_norm', float(backbone.router.structure_prior.grad.norm()))
        print('route_scores_shape', tuple(backbone.last_route_scores.shape))
        print('route_indices', backbone.last_route_indices.tolist())


if __name__ == '__main__':
    main()
