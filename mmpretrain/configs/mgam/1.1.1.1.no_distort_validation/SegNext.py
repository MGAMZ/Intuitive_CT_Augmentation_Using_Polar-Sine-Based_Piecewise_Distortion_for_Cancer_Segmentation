_base_ = './mgam.py'

model = dict(
    backbone=dict(
        type='mmseg.MSCAN',
        in_channels=1,
        norm_cfg = dict(type='BN', requires_grad=True),
        out_indices=[3],
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='SimpleRelativePositionHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=512,
            hid_channels=64,
            out_channels=1,
            with_avg_pool=False,
            with_last_bias=True,
        ),
        loss=dict(type='PixelReconstructionLoss',criterion='L1')
    ),
)
