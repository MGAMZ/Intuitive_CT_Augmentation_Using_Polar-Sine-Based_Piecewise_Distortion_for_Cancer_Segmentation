from mmengine.config import read_base
with read_base():
    from ..mgam_base import *

from mmseg.engine.hooks import SegVisualizationHook
from mgamdata.SA_Med2D import (\
    SA_Med2D_Dataset_NormalSample, 
    Load_SA_Med2D_SingleSlice, ForceResize,
    LabelSqueeze, ClassRectify, IoU_SupportsMultipleIgnoreIndex)
from mgamdata.mmeng_module import mgam_PerClassMetricLogger_OnTest
from mgamdata.DistortionAugment import LabelResize

# SETTING: SA-Med2D
modality = 'mr_t2'
dataset_source = 'BraTS2020'
activate_case_ratio = 1.0
num_slices_per_sample = 1
in_chans = 3

# SETTING: Distortion
distort_global_rotate = 0                 # 全局旋转角度，随机介于[-90x°, 90x°]
distort_amplitude = 0                     # 扭曲的振幅，沿中心向四周仿射的正弦振幅
distort_frequency = 1                     # 扭曲的频率，沿中心向四周仿射的正弦频率
distort_grid_dense = 36                   # 分段仿射映射的网格密度
distort_refresh_interval = 50             # 每隔多少个扭曲调用刷新一次映射矩阵

# SETTING: Normalization
norm_method = 'inst'                   # const:按照训练集分布归一化, inst:实例归一化。仅针对CTImageEnhance方法。hist:直方图均衡化。
HistEqual = True if norm_method=='hist' else False        # 是否进行直方图均衡化
HistEqualUseMask = True                   # 进行直方图均衡化是否基于mask

# SETTING: neural network
lr = 1e-4
iters = 20000 if not debug else 1

# SETTING: Binary Segmentation Mode
SingleChannelMode = False
num_classes = 2
threshold = 0.3
out_channels = num_classes
HeadUseSigmoid = False
HeadClassWeight = None




# --------------------PARAMETERS-------------------- #


# ////////////////////////////////////////////////// #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# --------------------CONPONENTS-------------------- #



# data preprocess
train_pipeline = [
    dict(type=Load_SA_Med2D_SingleSlice, 
         load_type=['image', 'label']),
    dict(type=ClassRectify),
    dict(type=LabelSqueeze),
    dict(type=ForceResize, image_size=crop_size),
    dict(type=LabelResize, size=crop_size),
    # dict(type=Distortion,
    #      global_rotate=distort_global_rotate,
    #      amplitude=distort_amplitude,
    #      frequency=distort_frequency,
    #      grid_dense=distort_grid_dense,
    #      in_array_shape=crop_size,
    #      refresh_interval=distort_refresh_interval),
    dict(type=RangeClipNorm, 
         input_shape=crop_size,
         norm_method=norm_method, 
         mean=mean, std=std),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=Load_SA_Med2D_SingleSlice, 
         load_type='image'),
    dict(type=Resize, scale=crop_size),
    dict(type=RangeClipNorm, 
         input_shape=crop_size,
         norm_method=norm_method, 
         mean=mean, std=std),
    dict(type=Load_SA_Med2D_SingleSlice,
         load_type='label'),
    dict(type=ClassRectify),
    dict(type=LabelSqueeze),
    dict(type=PackSegInputs)
]




train_dataset = dict(
    type=SA_Med2D_Dataset_NormalSample, 
    pipeline=train_pipeline,
    modality=modality,
    dataset_source=dataset_source,
    activate_case_ratio=activate_case_ratio,
    num_slices_per_sample=num_slices_per_sample,
    split='train',
    union_atom_rectify=True,
    debug=debug,
)
val_dataset = dict(
    type=SA_Med2D_Dataset_NormalSample, 
    pipeline=test_pipeline,
    modality=modality,
    dataset_source=dataset_source,
    activate_case_ratio=activate_case_ratio,
    num_slices_per_sample=num_slices_per_sample,
    split='val',
    union_atom_rectify=True,
    debug=debug,
)
test_dataset = dict(
    type=SA_Med2D_Dataset_NormalSample, 
    pipeline=test_pipeline,
    modality=modality,
    dataset_source=dataset_source,
    activate_case_ratio=activate_case_ratio,
    num_slices_per_sample=num_slices_per_sample,
    split='test',
    union_atom_rectify=True,
    debug=debug,
)

train_dataloader.update(dataset=train_dataset)
val_dataloader.update(dataset=val_dataset)
test_dataloader.update(dataset=test_dataset)

# data preprocessor
dpd.update(size=crop_size)
data_preprocessor.update(size=crop_size)

# train
train_cfg = dict(type=IterBasedTrainLoop, 
                 max_iters=iters, 
                 val_interval=val_interval)

# Optim
optim_wrapper.update(optimizer=dict(lr=lr))
param_scheduler = []

default_hooks.visualization=dict(type=SegVisualizationHook)
default_hooks.logger = dict(
    type=mgam_PerClassMetricLogger_OnTest, 
    interval=200, 
    log_metric_by_epoch=False)

val_evaluator.update(type=IoU_SupportsMultipleIgnoreIndex, prefix='Perf')
test_evaluator.update(type=IoU_SupportsMultipleIgnoreIndex, prefix='Perf')
