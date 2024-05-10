from mmengine.config import read_base
with read_base():
	from .mgam import *

from mmpretrain.models import PoolFormer
from mmseg.models import SegformerHead

PretrainedBackbone = '../TrainedModel/MMSeg.6.1.2.FineTuneWithRotation/Poolformer.pth'

model.update(dict(
	init_cfg=dict(type='Pretrained', checkpoint=PretrainedBackbone),
	backbone=dict(
		type=PoolFormer,
		arch='m48',
		in_chans=1,
		out_indices=(0,2,4,6),
	),
	decode_head=dict(
		type=SegformerHead,
		channels=1024,
		in_channels=[96,192,384,768],	# ConvNext atto:[40, 80, 160, 320] femto:[48, 96, 192, 384] pico:[64, 128, 256, 512] nano:[80, 160, 320, 640]; tiny-small:[96, 192, 384, 768]; base:[128, 256, 512, 1024]
		out_channels=out_channels,
		threshold=threshold,
		in_index=[0,1,2,3],
		num_classes=num_classes,
		norm_cfg = dict(type=BatchNorm2d, requires_grad=True),
		loss_decode=dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
					if SingleChannelMode else	\
					[dict(type=DiceLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
					dict(type=CrossEntropyLoss, use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
	),
))
