debug=False

# SETTING: database
dataset_type = "CQK_Med_Pretrain_Dataset"
data_root = r"C:\mgam\mgam_CT\2023_Med_CQK"
meta_ckpt = r"C:\mgam\mgam_CT\2023_Med_CQK\2024-02-05_16-06-11.pickle"
data_task = 'RelativeYLearning'
gt_field = 'normal'
samples_per_scan = 100 if not debug else 1
minimum_negative_distance = 20

# SETTING: data preprocess
contrast = 1                            # 全局对比度控制，影响标准差
brightness = 0                          # 全局亮度控制，影响均值
mean, std = 561.54, 486.59              # before clip to (0,4095): (10.87, 1138.37), after: (561.54, 486.59)
relative_position_std = 300             # 相对位置标准差
source_img_shape = (512,512)
norm_method = 'const'                   # const:按照训练集分布归一化, inst:实例归一化。仅针对CTImageEnhance方法。hist:直方图均衡化。
data_preprocessor_normalize = False     # 是否启用mmseg的data_preprocessor归一化（归一化可以在自定义的CTImageEnhance中进行）
distort_global_rotate=0                 # 全局旋转角度，随机介于[-90x°, 90x°]
distort_amplitude=2                     # 扭曲的振幅，沿中心向四周仿射的正弦振幅
distort_frequency=1.5                   # 扭曲的频率，沿中心向四周仿射的正弦频率
distort_grid_dense=15                   # 分段仿射映射的网格密度
distort_refresh_interval=20             # 每隔多少个扭曲调用刷新一次映射矩阵
HistEqualUseMask=True                   # 进行直方图均衡化是否基于mask
HistEqual=norm_method=='hist'           # 是否进行直方图均衡化
if norm_method == 'inst': assert data_preprocessor_normalize == False, \
    "Instance Norm 只能用CTImageEnhance实现，此时mmseg自带的预处理器应当禁用归一化"

# SETTING: neural network
crop_size = (256,256)
batch_size = 4 if not debug else 2
accumulative_counts=8
lr = 1e-4
workers = 4 if not debug else 0
iters = 10000 if not debug else 2
val_interval = 1000 if not debug else 1

# SETTING: Binary Segmentation Mode
SingleChannelMode = True
num_classes = 2
threshold = 0.3 if SingleChannelMode else None
out_channels = 1 if SingleChannelMode else 2
HeadUseSigmoid = True if SingleChannelMode else False
HeadClassWeight = None if SingleChannelMode else [0.1, 1]



train_pipeline = [
    dict(type="mmseg.LoadCTImage"),
    dict(type='Resize', scale=crop_size),
    dict(type='mmseg.Distortion',
         global_rotate=distort_global_rotate,
         amplitude=distort_amplitude,
         frequency=distort_frequency,
         grid_dense=distort_grid_dense,
         in_array_shape=crop_size,
         refresh_interval=distort_refresh_interval),
    dict(type='mmseg.RangeClipNorm', 
         contrast=contrast, 
         brightness=brightness, 
         norm_method=norm_method, 
         mean=mean, std=std),
    dict(type='mmseg.HistogramEqualization', 
              input_shape=crop_size, 
              use_mask=HistEqualUseMask,
              enabled=HistEqual),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type="mmseg.LoadCTImage"),
    dict(type='Resize', scale=crop_size),
    dict(type='mmseg.RangeClipNorm', 
         contrast=contrast, 
         brightness=brightness, 
         norm_method=norm_method, 
         mean=mean, std=std),
    dict(type='mmseg.HistogramEqualization', 
         input_shape=crop_size, 
         use_mask=HistEqualUseMask,
         enabled=HistEqual),
    dict(type='PackInputs')
]



# dataloader
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    pin_memory=True,
    persistent_workers=True if workers > 0 else False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    prefetch_factor= 4 if workers > 0 else None,
    dataset=dict(
        type=dataset_type,
        root = data_root,
        metadata_ckpt=meta_ckpt,
        task=data_task,
        gt_field=gt_field,
        split='train',
        samples_per_scan=samples_per_scan,
        minimum_negative_distance=minimum_negative_distance,
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers//2,
    pin_memory=True,
    persistent_workers=True if workers > 0 else False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        root = data_root,
        metadata_ckpt=meta_ckpt,
        task=data_task,
        gt_field=gt_field,
        split='val',
        samples_per_scan=samples_per_scan,
        minimum_negative_distance=minimum_negative_distance,
        pipeline=test_pipeline,
    )
)
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers//2,
    persistent_workers=True if workers > 0 else False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        root = data_root,
        metadata_ckpt=meta_ckpt,
        task=data_task,
        gt_field=gt_field,
        split='test',
        samples_per_scan=samples_per_scan,
        minimum_negative_distance=minimum_negative_distance,
        pipeline=test_pipeline,
    )
)

val_evaluator = dict(type='RelativePositionError', loss_type='L1')
# val_evaluator = dict(type='Accuracy', topk=(4, 16, 64, 128))
test_evaluator = val_evaluator


# data preprocessor dict
dpd = data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=mean if data_preprocessor_normalize else None,
    std=std if data_preprocessor_normalize else None,
    pad_value=0,
    non_blocking=True,
)

model = dict(type='mmpretrain.RelativePositionLearning',
             label_std=relative_position_std)

# optimizer and scheduler
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr),
    clip_grad=dict(type='norm', max_norm=1e6, norm_type=2, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-2, begin=0, end=iters*0.1,
        by_epoch=False),
    dict(
        type='PolyLR',
        eta_min=1e-2*lr,
        power=0.3,
        begin=iters*0.2,
        end=iters,
        by_epoch=False)
]



# Task Control
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)



# runtime env
default_scope = 'mmpretrain'
runner_type='mgam_Runner'
env_cfg = dict(
    cudnn_benchmark=True,
    cuda_matmul_allow_tf32=True,
    cudnn_allow_tf32=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
work_dir = './work_dirs/'
randomness = dict(seed=None, deterministic=False)
