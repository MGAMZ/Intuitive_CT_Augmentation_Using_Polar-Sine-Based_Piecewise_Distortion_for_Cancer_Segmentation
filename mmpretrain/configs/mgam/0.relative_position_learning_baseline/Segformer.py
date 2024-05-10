_base_ = './mgam.py'


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmseg.MixVisionTransformer',
        in_channels=1,
        out_indices=[3],
        embed_dims=192,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=500,
        in_channels=1536,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
)


