_base_ = './mgam.py'


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmseg.ResNetV1c',
        depth=50,
        in_channels=1,
        out_indices=[3],
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=500,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
)



