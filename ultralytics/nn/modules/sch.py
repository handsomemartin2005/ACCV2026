import torch

from .head import GUIDE_RTDETRDecoder, UnifiedHypergraphReasoning

__all__ = 'SCH_GUIDE_RTDETRDecoder',


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
