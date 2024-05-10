_base_ = './mgam.py'

model = dict(
    backbone=dict(
        type='SwinTransformerV2',
        arch='base',
        img_size=256,
        in_channels=1,
        out_indices=[3],
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SimpleRelativePositionHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=1024,
            hid_channels=64,
            out_channels=1,
            with_avg_pool=False,
            with_last_bias=True,
        ),
        loss=dict(type='PixelReconstructionLoss',criterion='L1')
    ),
)

