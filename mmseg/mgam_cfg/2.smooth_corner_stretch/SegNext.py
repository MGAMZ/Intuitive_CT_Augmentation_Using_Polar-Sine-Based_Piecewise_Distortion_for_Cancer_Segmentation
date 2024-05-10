_base_ = './mgam.py'

out_channels = {{_base_.out_channels}}
num_classes = {{_base_.num_classes}}
threshold = {{_base_.threshold}}
HeadUseSigmoid = {{_base_.HeadUseSigmoid}}
HeadClassWeight = {{_base_.HeadClassWeight}}
SingleChannelMode = {{_base_.SingleChannelMode}}
crop_size = {{_base_.crop_size}}
dpd = {{_base_.dpd}}
workers = 20


# 模型比较小, 瓶颈位于CPU
train_dataloader = dict(num_workers=workers)
val_dataloader = dict(num_workers=workers//2)
test_dataloader = dict(num_workers=workers//2)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dpd,
    backbone=dict(
        type='MSCAN',
        in_channels=1,
        norm_cfg = dict(type='BN', requires_grad=True),
    ),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 256, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        out_channels=out_channels,
        threshold=threshold,
        num_classes=num_classes,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=HeadUseSigmoid, loss_weight=1.0, class_weight=HeadClassWeight)	\
                    if SingleChannelMode else	\
                    [dict(type='DiceLoss', use_sigmoid=HeadUseSigmoid, loss_weight=0.7),	\
                    dict(type='CrossEntropyLoss', use_sigmoid=HeadUseSigmoid, loss_weight=0.3, class_weight=HeadClassWeight)],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

