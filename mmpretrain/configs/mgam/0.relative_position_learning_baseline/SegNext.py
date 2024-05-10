_base_ = './mgam.py'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmseg.MSCAN',
        in_channels=1,
        norm_cfg = dict(type='BN', requires_grad=True),
        out_indices=[3],
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=500,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
)
