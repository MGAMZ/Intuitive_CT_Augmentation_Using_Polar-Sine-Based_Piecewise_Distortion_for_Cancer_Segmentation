_base_ = 'mgam.py'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='base',
        in_channels=1,
        out_indices=-1,
        use_grn=True,
        gap_before_final_norm=True,
        ),
    head=dict(
        type='LinearClsHead',
        num_classes=500,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
)

