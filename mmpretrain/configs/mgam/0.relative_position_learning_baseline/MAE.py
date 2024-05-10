_base_ = './mgam.py'

crop_size = _base_.crop_size

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmseg.MAE',
        in_channels=1,
        img_size=crop_size,
        patch_size=8,
        num_heads=8,
        embed_dims=384,
        out_indices=-1,
    ),
	neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=500,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
)
