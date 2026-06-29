from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'ultralytics/cfg/models/rt-detr/ablations'


HEAD = """head:
  - [-1, 1, Conv, [256, 1, 1, None, 1, 1, False]]
  - [-1, 1, AIFI, [1024, 8]]
  - [-1, 1, Conv, [256, 1, 1]]

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [3, 1, Conv, [256, 1, 1, None, 1, 1, False]]
  - [[-2, -1], 1, Concat, [1]]
  - [-1, 3, RepC3, [256, 0.5]]
  - [-1, 1, Conv, [256, 1, 1]]

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [2, 1, Conv, [256, 1, 1, None, 1, 1, False]]
  - [[-2, -1], 1, Concat, [1]]
  - [-1, 3, RepC3, [256, 0.5]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, RepC3, [256, 0.5]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 7], 1, Concat, [1]]
  - [-1, 3, RepC3, [256, 0.5]]

"""


def yaml_bool(value):
    return 'true' if value else 'false'


def backbone_args(candidate_scales=5, topk=3, use_clip=True, output_scales=3,
                  use_semantic_prior=True, use_high_frequency=True,
                  use_scale_prior=True, use_structure_prior=True,
                  use_guide_semantic=True, use_clip_image=True):
    values = [
        candidate_scales,
        topk,
        'RN50',
        'openai',
        yaml_bool(use_clip),
        output_scales,
        yaml_bool(use_semantic_prior),
        yaml_bool(use_high_frequency),
        yaml_bool(use_scale_prior),
        yaml_bool(use_structure_prior),
        yaml_bool(use_guide_semantic),
        yaml_bool(use_clip_image),
    ]
    return '[' + ', '.join(map(str, values)) + ']'


def make_config(name, args, decoder='SCH_GUIDE_RTDETRDecoder', uhr_cfg=None):
    uhr_suffix = ''
    if uhr_cfg:
        pairs = ', '.join(f'{k}: {yaml_bool(v)}' for k, v in uhr_cfg.items())
        uhr_suffix = f', {{{pairs}}}'
    return f"""# Auto-generated SCH-MDETR ablation: {name}

nc: 17
scales:
  l: [1.00, 1.00, 1024]

backbone:
  - [-1, 1, sch_mamba_adapters, {args}]

{HEAD}  - [[1, 16, 19, 22], 1, {decoder}, [nc, 192, 300, 4, 8, 3{uhr_suffix}]]
"""


EXPERIMENTS = {
    'sch_dtsr_k3': make_config(
        'DTSR only, K=3',
        backbone_args(use_clip=False, use_semantic_prior=False, use_guide_semantic=False),
        decoder='GUIDE_RTDETRDecoder'),
    'sch_clip_dtsr_k3': make_config(
        'CLIP + DTSR, K=3, no UHR',
        backbone_args(use_clip=True),
        decoder='GUIDE_RTDETRDecoder'),
    'sch_dtsr_uhr_k3': make_config(
        'DTSR + UHR, K=3, no CLIP',
        backbone_args(use_clip=False, use_semantic_prior=False, use_guide_semantic=False)),
    'sch_full_k1': make_config('Full SCH, K=1', backbone_args(topk=1)),
    'sch_full_k2': make_config('Full SCH, K=2', backbone_args(topk=2)),
    'sch_full_k3': make_config('Full SCH, K=3', backbone_args(topk=3)),
    'sch_full_k4': make_config('Full SCH, K=4', backbone_args(topk=4)),
    'sch_full_m4_k3': make_config('Full SCH, M=4, K=3', backbone_args(candidate_scales=4, topk=3)),
    'sch_full_m5_k3': make_config('Full SCH, M=5, K=3', backbone_args(candidate_scales=5, topk=3)),
    'sch_full_m6_k3': make_config('Full SCH, M=6, K=3', backbone_args(candidate_scales=6, topk=3)),
    'sch_no_highfreq_k3': make_config(
        'Full SCH without high-frequency router prior',
        backbone_args(use_high_frequency=False)),
    'sch_no_scale_prior_k3': make_config(
        'Full SCH without scale-statistic/scale-embedding router prior',
        backbone_args(use_scale_prior=False)),
    'sch_no_structure_prior_k3': make_config(
        'Full SCH without router structure prior',
        backbone_args(use_structure_prior=False)),
    'sch_clip_text_only_k3': make_config(
        'Full SCH with CLIP text prototype only',
        backbone_args(use_clip_image=False)),
    'sch_uhr_no_category_k3': make_config(
        'Full SCH without UHR category branch',
        backbone_args(),
        uhr_cfg={'use_category': False, 'use_bbox': True, 'use_structure': True}),
    'sch_uhr_no_bbox_k3': make_config(
        'Full SCH without UHR bbox branch',
        backbone_args(),
        uhr_cfg={'use_category': True, 'use_bbox': False, 'use_structure': True}),
    'sch_uhr_no_structure_k3': make_config(
        'Full SCH without UHR structure branch',
        backbone_args(),
        uhr_cfg={'use_category': True, 'use_bbox': True, 'use_structure': False}),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in EXPERIMENTS.items():
        path = OUT_DIR / f'{name}.yaml'
        path.write_text(text, encoding='utf-8')
        print(path)


if __name__ == '__main__':
    main()
