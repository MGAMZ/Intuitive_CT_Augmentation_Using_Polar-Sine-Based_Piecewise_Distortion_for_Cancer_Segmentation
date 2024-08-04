from mmengine.config import read_base
with read_base():
	from .mgam import *

from mmseg.models import MAE
from mmseg.models import Feature2Pyramid
from mmseg.models import SegformerHead


model.update(dict(
	backbone=dict(
		type=MAE,
		in_channels=in_chans,
		img_size=crop_size,
		patch_size=8,
		num_heads=8,
		embed_dims=384,
		out_indices=(3,5,7,11),
	),
	neck=dict(type=Feature2Pyramid, embed_dim=384, rescales=[4, 2, 1, 0.5]),
	decode_head=dict(
		type=SegformerHead,
		in_channels=[384,384,384,384],
		in_index=[0,1,2,3],
		channels=384,
		out_channels=num_classes,
		threshold=threshold,
		num_classes=num_classes,
		norm_cfg=dict(type=GroupNorm, num_groups=32, requires_grad=True),
		loss_decode=dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
					if SingleChannelMode else	\
					[dict(type=DiceLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
					dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
	),
))
