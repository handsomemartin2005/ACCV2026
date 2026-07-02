import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import GUIDE_RTDETRDecoder, RTDETRDecoder, UnifiedHypergraphReasoning

__all__ = 'SCH_GUIDE_RTDETRDecoder', 'LightSCH_RTDETRDecoder'


class SCH_GUIDE_RTDETRDecoder(GUIDE_RTDETRDecoder):
    """SCH-MDETR decoder with CLIP-style prototypes and unified hypergraph reasoning."""

    def __init__(self, *args, **kwargs):
        uhr_cfg = {}
        if args and isinstance(args[-1], dict):
            *args, uhr_cfg = args
        uhr_cfg.update(kwargs.pop('uhr_cfg', {}))
        super().__init__(*args, **kwargs)
        self.uhr = UnifiedHypergraphReasoning(self.hidden_dim, self.nc, **uhr_cfg)

    def forward(self, x, batch=None):
        from ultralytics.models.utils.ops import get_cdn_group

        backbone_feats = x[0]
        feats, shapes = self._get_encoder_input(x[1:])

        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        embed, refer_bbox = self.uhr(embed, refer_bbox, backbone_feats)

        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            backbone_feats,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)


class LightTextSemanticPrior(nn.Module):
    """Text-only semantic prior for lightweight SCH variants.

    CLIP text features are encoded once at initialization and stored as a buffer,
    so training does not carry a CLIP image/text tower in the forward pass.
    """

    def __init__(self, hidden_dim, category_names=None, prompt_template='a visible-light image containing a {name}',
                 clip_model='RN50', clip_pretrained='openai', use_clip_text=True):
        super().__init__()
        self.category_names = tuple(category_names or ('UAV',))
        self.use_clip_text = False
        clip_dim = 512
        if use_clip_text:
            try:
                import open_clip
                clip, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained, device='cpu')
                tokenizer = open_clip.get_tokenizer(clip_model)
                prompts = [prompt_template.format(name=name) for name in self.category_names]
                tokens = tokenizer(prompts)
                clip.eval()
                with torch.no_grad():
                    text_features = clip.encode_text(tokens).float()
                text_features = F.normalize(text_features, dim=-1)
                clip_dim = text_features.shape[-1]
                self.register_buffer('clip_text_features', text_features, persistent=False)
                self.use_clip_text = True
                del clip
            except Exception:
                self.register_buffer('clip_text_features', torch.empty(0), persistent=False)
        else:
            self.register_buffer('clip_text_features', torch.empty(0), persistent=False)
        self.clip_to_model = nn.Linear(clip_dim, hidden_dim)
        self.fallback_category_prototypes = nn.Parameter(torch.randn(len(self.category_names), hidden_dim) * 0.02)
        self.visual_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, projected_feats):
        pooled = torch.stack([F.adaptive_avg_pool2d(feat, 1).flatten(1) for feat in projected_feats], dim=1).mean(1)
        visual_prior = F.normalize(self.visual_proj(pooled), dim=-1)
        if self.use_clip_text and self.clip_text_features.numel():
            prototypes = self.clip_to_model(self.clip_text_features.to(pooled.device, pooled.dtype))
            prototypes = F.normalize(prototypes, dim=-1)
        else:
            prototypes = F.normalize(self.fallback_category_prototypes, dim=-1)
        semantic_response = visual_prior @ prototypes.t()
        semantic_prior = semantic_response.softmax(dim=-1) @ prototypes
        return semantic_prior, semantic_response


class LightDynamicTopKScaleRouter(nn.Module):
    """Image-aware TopK scale router that keeps RT-DETR-N tensor shapes stable."""

    def __init__(self, candidate_ch, hidden_dim=128, topk=3, use_semantic_prior=True,
                 use_high_frequency=True, use_scale_prior=True, use_structure_prior=True):
        super().__init__()
        self.topk = topk
        self.use_semantic_prior = use_semantic_prior
        self.use_high_frequency = use_high_frequency
        self.use_scale_prior = use_scale_prior
        self.use_structure_prior = use_structure_prior
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(ch, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim))
            for ch in candidate_ch
        )
        mlp_hidden = max(hidden_dim // 2, 32)
        self.scale_embed = nn.Parameter(torch.randn(len(candidate_ch), hidden_dim) * 0.02)
        self.structure_prior = nn.Parameter(torch.zeros(hidden_dim))
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1)
        )

    @staticmethod
    def _resize_like(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _high_frequency_score(x):
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return (x - smooth).abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

    def score(self, candidates, semantic_prior):
        scores = []
        b = candidates[0].shape[0]
        for i, feat in enumerate(candidates):
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            semantic = semantic_prior if self.use_semantic_prior else torch.zeros_like(semantic_prior)
            high_freq = self._high_frequency_score(feat) if self.use_high_frequency else feat.new_zeros((b, 1))
            h, w = feat.shape[-2:]
            scale_value = feat.new_full((b, 1), math.log(max(h * w, 1))) if self.use_scale_prior else feat.new_zeros((b, 1))
            scale_prior = self.scale_embed[i].to(feat.device, feat.dtype).expand(b, -1) if self.use_scale_prior else torch.zeros_like(pooled)
            structure_prior = self.structure_prior.to(feat.device, feat.dtype).expand(b, -1) if self.use_structure_prior else torch.zeros_like(pooled)
            route_in = torch.cat((pooled, semantic, scale_prior, structure_prior, high_freq, scale_value), dim=1)
            scores.append(self.router(route_in))
        return torch.cat(scores, dim=1)

    def route_projected(self, candidates, semantic_prior, target_refs):
        scores = self.score(candidates, semantic_prior)
        b = candidates[0].shape[0]
        k = min(max(self.topk, 1), len(candidates))
        selected_idx = torch.topk(scores.mean(0), k=k).indices.sort().values.tolist()
        topk_weights = scores[:, selected_idx].softmax(dim=1)
        all_weights = scores.softmax(dim=1)
        routed = []
        for ref in target_refs:
            hard = sum(
                self._resize_like(candidates[idx], ref) * topk_weights[:, j].view(b, 1, 1, 1)
                for j, idx in enumerate(selected_idx)
            )
            if self.training:
                soft = sum(
                    self._resize_like(candidate, ref) * all_weights[:, j].view(b, 1, 1, 1)
                    for j, candidate in enumerate(candidates)
                )
                hard = hard + soft - soft.detach()
            routed.append(hard)
        return candidates, routed, scores, selected_idx

    def forward(self, raw_feats, semantic_prior, target_refs):
        candidates = [proj(feat) for proj, feat in zip(self.proj, raw_feats)]
        return self.route_projected(candidates, semantic_prior, target_refs)


class LightSCH_RTDETRDecoder(RTDETRDecoder):
    """RT-DETR-N decoder with lightweight SCH routing and UHR."""

    def __init__(self, nc=1, candidate_ch=(64, 128, 128, 128), hd=128, nq=100, ndp=4, nh=4, ndl=3,
                 d_ffn=512, topk=3, use_clip_text=True, use_semantic_prior=True, use_high_frequency=True,
                 use_scale_prior=True, use_structure_prior=True, category_names=None,
                 prompt_template='a visible-light image containing a {name}', clip_model='RN50',
                 clip_pretrained='openai', uhr_cfg=None):
        super().__init__(nc, [hd] * topk, hd, nq, ndp, nh, ndl, d_ffn)
        self.topk = topk
        self.semantic_prior = LightTextSemanticPrior(
            hd,
            category_names=category_names or ('UAV',),
            prompt_template=prompt_template,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            use_clip_text=use_clip_text
        )
        self.scale_router = LightDynamicTopKScaleRouter(
            candidate_ch,
            hidden_dim=hd,
            topk=topk,
            use_semantic_prior=use_semantic_prior,
            use_high_frequency=use_high_frequency,
            use_scale_prior=use_scale_prior,
            use_structure_prior=use_structure_prior
        )
        self.uhr = UnifiedHypergraphReasoning(hd, nc, **(uhr_cfg or {}))

    def forward(self, x, batch=None):
        from ultralytics.models.utils.ops import get_cdn_group

        raw_feats = list(x)
        target_refs = raw_feats[-self.topk:]
        projected = [proj(feat) for proj, feat in zip(self.scale_router.proj, raw_feats)]
        semantic_prior, semantic_response = self.semantic_prior(projected)
        candidates, routed_feats, route_scores, selected_idx = self.scale_router.route_projected(
            projected, semantic_prior, target_refs)

        self.last_route_scores = route_scores.detach()
        self.last_route_gates = route_scores.sigmoid().detach()
        self.last_route_indices = torch.as_tensor(selected_idx, device=route_scores.device)
        self.last_candidate_shapes = [tuple(feat.shape[-2:]) for feat in candidates]
        self.last_semantic_response = semantic_response.detach()

        feats, shapes = self._get_encoder_input(routed_feats)

        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        structure_feat = routed_feats[0]
        embed, refer_bbox = self.uhr(embed, refer_bbox, structure_feat)

        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)
