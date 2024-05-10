_base_ = './mgam.py'

out_channels = _base_.out_channels
num_classes = _base_.num_classes
threshold = _base_.threshold
HeadUseSigmoid = _base_.HeadUseSigmoid
HeadClassWeight = _base_.HeadClassWeight
SingleChannelMode = _base_.SingleChannelMode
crop_size = _base_.crop_size
dpd = _base_.dpd


model = dict(
    type='EncoderDecoder',
    data_preprocessor=dpd,
    backbone=dict(
        type='MAE',
        in_channels=1,
        img_size=crop_size,
        patch_size=8,
        num_heads=8,
        embed_dims=384,
        out_indices=(3,5,7,11),
    ),
    neck=dict(type='Feature2Pyramid', embed_dim=384, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[384,384,384,384],
        in_index=[0,1,2,3],
        channels=384,
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
