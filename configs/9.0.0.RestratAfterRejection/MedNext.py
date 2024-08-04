from mmengine.config import read_base
with read_base():
    from .mgam import *

from mgamdata.models.mednext.MedNextV1 import MM_MedNext_Encoder, MM_MedNext_Decoder


compile.update(dynamic=False)

model.update(dict(
    backbone=dict(
        type=MM_MedNext_Encoder,
        in_channels=1,
        embed_dims=32,
    ),
    decode_head=dict(
        type=MM_MedNext_Decoder,
        embed_dims=32,
        num_classes=num_classes,
        out_channels=out_channels,
        threshold=threshold,
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