_base_ = 'mgam.py'

model = dict(
    backbone=dict(
        type='ConvNeXt',
        arch='small',
        in_channels=1,
        out_indices=-1,
        use_grn=True,
        gap_before_final_norm=False,
        ),
    head=dict(
        type='SimpleRelativePositionHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=768,
            hid_channels=64,
            out_channels=1,
            with_avg_pool=True,
            with_last_bias=True,
        ),
        loss=dict(type='PixelReconstructionLoss',criterion='L1')
    ),
)

