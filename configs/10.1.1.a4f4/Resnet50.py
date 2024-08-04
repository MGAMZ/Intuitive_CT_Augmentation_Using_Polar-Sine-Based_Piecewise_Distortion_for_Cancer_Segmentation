from mmengine.config import read_base
with read_base():
	from .mgam import *

from mmseg.models import ResNetV1c
from mmseg.models import UPerHead
from mmseg.models import FCNHead


model.update(dict(
	backbone=dict(
		type=ResNetV1c,
		depth=50,
		in_channels=in_chans,
	),
	decode_head=dict(
		type=UPerHead,
		channels=1024,
		in_channels=[256, 512, 1024, 2048],
		out_channels=num_classes,
		threshold=threshold,
		in_index=[0,1,2,3],
		num_classes=num_classes,
		norm_cfg = dict(type=BatchNorm2d, requires_grad=True),
		align_corners=False,
		loss_decode=dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
					if SingleChannelMode else	\
					[dict(type=DiceLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
					dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
	),
))

