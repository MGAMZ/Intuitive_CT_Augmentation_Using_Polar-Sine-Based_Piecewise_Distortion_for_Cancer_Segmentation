from mmengine.config import read_base
with read_base():
	from .mgam import *

from mmseg.models import MSCAN
from mmseg.models import LightHamHead

# MMSEG SegNext预训练加载权重后下游任务无法学习，原因未知，暂时跳过预训练。
PretrainedBackbone = '../mmpretrain/work_dirs/2.2.1.boost/round_1/SegNext/iter_100000.pth'

model.update(dict(
	backbone=dict(
		type=MSCAN,
		in_channels=1,
		norm_cfg = dict(type=BatchNorm2d, requires_grad=True),
		init_cfg=dict(type='Pretrained', checkpoint=PretrainedBackbone, prefix='backbone.')
	),
	decode_head=dict(
		type=LightHamHead,
		in_channels=[128, 256, 512],
		in_index=[1, 2, 3],
		channels=512,
		ham_channels=512,
		out_channels=out_channels,
		threshold=threshold,
		num_classes=num_classes,
		norm_cfg=dict(type=GroupNorm, num_groups=32, requires_grad=True),
		loss_decode=dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
					if SingleChannelMode else	\
					[dict(type=DiceLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
					dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
	)
))

