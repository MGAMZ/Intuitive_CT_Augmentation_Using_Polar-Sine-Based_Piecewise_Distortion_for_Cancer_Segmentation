from mmengine.config import read_base
with read_base():
	from .mgam import *
	
from mmpretrain.models.backbones import ConvNeXt
from mmpretrain.models.heads import SimpleRelativePositionHead
from mmpretrain.models.necks import NonLinearNeck
from mmpretrain.models.losses import PixelReconstructionLoss

model.update(dict(
	backbone=dict(
		type=ConvNeXt,
		arch='base',
		in_channels=1,
		out_indices=-1,
		use_grn=True,
		gap_before_final_norm=False,
		),
	head=dict(
		type=SimpleRelativePositionHead,
		predictor=dict(
			type=NonLinearNeck,
			in_channels=1024,
			hid_channels=64,
			out_channels=1,
			with_avg_pool=True,
			with_last_bias=True,
		),
		loss=dict(type=PixelReconstructionLoss,criterion='L1')
	),
))

