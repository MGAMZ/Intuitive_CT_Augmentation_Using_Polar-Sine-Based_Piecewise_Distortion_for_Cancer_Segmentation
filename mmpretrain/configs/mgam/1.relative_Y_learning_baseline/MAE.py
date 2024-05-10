_base_ = './mgam.py'

crop_size = _base_.crop_size

model = dict(
    backbone=dict(
        type='mmseg.MAE',
        in_channels=1,
        img_size=crop_size,
        patch_size=16,
        num_heads=8,
        embed_dims=256,
        out_indices=-1,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SimpleRelativePositionHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=64,
            out_channels=1,
            with_avg_pool=False,
            with_last_bias=True,
        ),
        loss=dict(type='PixelReconstructionLoss',criterion='L1')
    ),
)
