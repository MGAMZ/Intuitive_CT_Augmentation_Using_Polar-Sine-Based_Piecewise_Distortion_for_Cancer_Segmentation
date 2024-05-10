_base_ = './mgam.py'
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

out_channels = {{_base_.out_channels}}
num_classes = {{_base_.num_classes}}
threshold = {{_base_.threshold}}
HeadUseSigmoid = {{_base_.HeadUseSigmoid}}
HeadClassWeight = {{_base_.HeadClassWeight}}
SingleChannelMode = {{_base_.SingleChannelMode}}
crop_size = {{_base_.crop_size}}
dpd = {{_base_.dpd}}


# 模型比较小, 瓶颈位于CPU
train_dataloader = dict(num_workers=20 if not {{_base_.debug}} else 0)
val_dataloader = dict(num_workers=12 if not {{_base_.debug}} else 0)
test_dataloader = dict(num_workers=12 if not {{_base_.debug}} else 0)

model = dict(
	type='EncoderDecoder',
	data_preprocessor=dpd,
	backbone=dict(
		type='mmpretrain.PoolFormer',
		arch='m48',
		in_chans=1,
		out_indices=(0,2,4,6),
	),
	decode_head=dict(
		type='SegformerHead',
		channels=1024,
		in_channels=[96,192,384,768],	# ConvNext atto:[40, 80, 160, 320] femto:[48, 96, 192, 384] pico:[64, 128, 256, 512] nano:[80, 160, 320, 640]; tiny-small:[96, 192, 384, 768]; base:[128, 256, 512, 1024]
		out_channels=out_channels,
		threshold=threshold,
		in_index=[0,1,2,3],
		num_classes=num_classes,
		norm_cfg = dict(type='BN', requires_grad=True),
		loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
					if SingleChannelMode else	\
					[dict(type='DiceLoss', use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
					dict(type='CrossEntropyLoss', use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
	),
	train_cfg=dict(),
	test_cfg=dict(mode='whole'),
)
