_base_ = './mgam.py'

out_channels = {{_base_.out_channels}}
num_classes = {{_base_.num_classes}}
threshold = {{_base_.threshold}}
HeadUseSigmoid = {{_base_.HeadUseSigmoid}}
HeadClassWeight = {{_base_.HeadClassWeight}}
SingleChannelMode = {{_base_.SingleChannelMode}}
crop_size = {{_base_.crop_size}}
dpd = {{_base_.dpd}}


model = dict(
	type='EncoderDecoder',
	data_preprocessor=dpd,
	backbone=dict(
		type='ResNetV1c',
		depth=50,
		in_channels=1,
	),
	decode_head=dict(
		type='UPerHead',
		channels=1024,
		in_channels=[256, 512, 1024, 2048],
		out_channels=out_channels,
		threshold=threshold,
		in_index=[0,1,2,3],
		num_classes=num_classes,
		norm_cfg = dict(type='BN', requires_grad=True),
		align_corners=False,
		loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
					if SingleChannelMode else	\
					[dict(type='DiceLoss', use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
					dict(type='CrossEntropyLoss', use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
	),
	auxiliary_head=dict(
		type='FCNHead',
		in_channels=1024,
		out_channels=out_channels,
		threshold=threshold,
		in_index=2,
		channels=1024,
		num_convs=2,
		concat_input=False,
		num_classes=num_classes,
		norm_cfg = dict(type='BN', requires_grad=True),
		align_corners=False,
		loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
					if SingleChannelMode else	\
					[dict(type='DiceLoss', use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
					dict(type='CrossEntropyLoss', use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
	),
	train_cfg=dict(),
	test_cfg=dict(mode='whole'),
)

