# dataset
dataset_type = "CQK_2023_Med"
data_root = r"D:\PostGraduate\DL\mgam_CT\data\2023_Med_CQK"
num_positive_img = 1
num_negative_img = 0
minimum_negative_distance = 50

# data preprocess
contrast = 1
brightness = -1000
norm_method = 'const'	# const:按照训练集分布归一化, inst:实例归一化
data_preprocessor_normalize = True
data_preprocessor_mean = [10.87] if data_preprocessor_normalize else None
data_preprocessor_std = [1138.37] if data_preprocessor_normalize else None

# model
crop_size = (256,256)
batch_size = 8
lr = 1e-4
workers = 2
iters = 20000
val_interval = 1000


# Binary Segmentation Mode
SingleChannelMode = True
num_classes = 2
threshold = 0.3 if SingleChannelMode else None
out_channels = 1 if SingleChannelMode else 2
HeadUseSigmoid = True if SingleChannelMode else False
HeadClassWeight = None if SingleChannelMode else [0.1, 1]



train_pipeline = [
    dict(type="LoadCTImage"),
    dict(type='LoadCTLabel'),
    dict(type='Resize', scale=crop_size, keep_ratio=True, interpolation='nearest'),
    dict(type='RandomRotate', prob=1, degree=(-90,90), pad_val=-2000, seg_pad_val=0),
    # dict(type='CTImgEnhance', contrast=contrast, brightness=brightness, norm_method=norm_method),	# CT影像 纵隔部位加强
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type="LoadCTImage"),
    dict(type='Resize', scale=crop_size, keep_ratio=True, interpolation='nearest'),
    # dict(type='CTImgEnhance', contrast=contrast, brightness=brightness, norm_method=norm_method),	# CT影像 纵隔部位加强
    dict(type='LoadCTLabel'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        PORT_ARGS=dict(
            root=data_root, 
            metadata_ckpt=r"D:\PostGraduate\DL\ClassicDataset\2023_Med_CQK\2023-11-26_14-23-05.pickle",
            split='train',
            pretraining=False,
            num_positive_img=num_positive_img,
            num_negative_img=num_negative_img,
            minimum_negative_distance=minimum_negative_distance,
            ensambled_img_group=False,
        ),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers*2,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, 
        PORT_ARGS=dict(
            root=data_root, 
            metadata_ckpt=r"D:\PostGraduate\DL\ClassicDataset\2023_Med_CQK\2023-11-26_14-23-05.pickle",
            split='val',
            pretraining=False,
            num_positive_img=num_positive_img,
            num_negative_img=num_negative_img,
            minimum_negative_distance=minimum_negative_distance,
            ensambled_img_group=False,
        ),
        pipeline=test_pipeline
    )
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, 
        PORT_ARGS=dict(
            root=data_root, 
            metadata_ckpt=r"D:\PostGraduate\DL\ClassicDataset\2023_Med_CQK\2023-11-26_14-23-05.pickle",
            split='test',
            pretraining=False,
            num_positive_img=num_positive_img,
            num_negative_img=num_negative_img,
            minimum_negative_distance=minimum_negative_distance,
            ensambled_img_group=False,
        ),
        pipeline=test_pipeline
    )
)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'], nan_to_num=0)
test_evaluator = val_evaluator


# model
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=data_preprocessor_mean, std=data_preprocessor_std,
    pad_val=0,
    seg_pad_val=0,
    non_blocking=True,
)


model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(type='SegDataPreProcessor', non_blocking=True, size=crop_size, mean=data_preprocessor_mean, std=data_preprocessor_std),
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


# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr),
    clip_grad=dict(type='norm', max_norm=1, norm_type=2, error_if_nonfinite=False)
)


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-2, begin=0, end=iters*0.1,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=1e-2*lr,
        power=0.6,
        begin=iters*0.2,
        end=iters,
        by_epoch=False)
]


train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=10, npy_read=True)
)


# runtime
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=0.4)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = True
work_dir = './work_dirs/BASELINE_MAE'
tta_model = None







