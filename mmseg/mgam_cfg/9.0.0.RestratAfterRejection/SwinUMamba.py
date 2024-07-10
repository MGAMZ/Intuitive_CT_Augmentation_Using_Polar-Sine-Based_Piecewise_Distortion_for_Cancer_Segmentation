from mmengine.config import read_base
with read_base():
    from .mgam import *

from mgamdata.models.SwinUMamba import MM_SwinUMamba_backbone, MM_SwinUMamba_decoder


compile.update(disable=True)    # compile将导致模型发散
layer_dims=[96, 192, 384, 768]

model.update(dict(
    # init_cfg=dict(type='Pretrained', checkpoint=PretrainedBackbone),
    backbone=dict(
        type=MM_SwinUMamba_backbone,
        in_chans=3, 
        patch_size=4, 
        depths=[2, 2, 9, 2], 
        dims=layer_dims, 
        d_state=16
    ),
    decode_head=dict(
        type=MM_SwinUMamba_decoder,
        channels=512,
        in_channels=layer_dims,
        out_channels=out_channels,
        threshold=threshold,
        in_index=[0,1,2,3],
        num_classes=num_classes,
        norm_cfg=None,
        align_corners=False,
        ignore_index=0,
        loss_decode=dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
                    if SingleChannelMode else	\
                    [dict(type=DiceLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
                    dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
    ),
    auxiliary_head=None,
))

