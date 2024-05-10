from mmengine.config import read_base
with read_base():
	from .mgam import *

	
from mmseg.models.backbones import MAE
from mmpretrain.models.heads import SimpleRelativePositionHead
from mmpretrain.models.necks import GlobalAveragePooling
from mmpretrain.models.necks import NonLinearNeck
from mmpretrain.models.losses import PixelReconstructionLoss

model.update(dict(
    backbone=dict(
        type=MAE,
        in_channels=1,
        img_size=crop_size,
        patch_size=8,
        num_heads=8,
        embed_dims=384,
        out_indices=-1,
    ),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=SimpleRelativePositionHead,
        predictor=dict(
            type=NonLinearNeck,
            in_channels=384,
            hid_channels=64,
            out_channels=1,
            with_avg_pool=False,
            with_last_bias=True,
        ),
        loss=dict(type=PixelReconstructionLoss,criterion='L1')
    ),
))
