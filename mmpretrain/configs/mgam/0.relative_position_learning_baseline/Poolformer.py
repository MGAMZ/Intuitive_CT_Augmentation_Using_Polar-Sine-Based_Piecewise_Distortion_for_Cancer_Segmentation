_base_ = './mgam.py'

model = dict(
    type='ImageClassifier',
	backbone=dict(
		type='PoolFormer',
		arch='m48',
		in_chans=1,
		out_indices=(6),
	),
	neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=500,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
)
