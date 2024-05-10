from mmengine.config import read_base
with read_base():
	from ..mgam_base import *
	
from mmpretrain.mgam_utils import RelativePositionLearning
from mmpretrain.mgam_utils import RelativePositionLearningDataset
from mmpretrain.mgam_utils import RelativePositionError
from mmpretrain.mgam_utils import MMPre_LoadCTImage

# SETTING: Normalization
contrast = 1                            # 全局对比度控制，影响标准差
brightness = 0                          # 全局亮度控制，影响均值
norm_method = 'const'                   # const:按照训练集分布归一化, inst:实例归一化。仅针对CTImageEnhance方法。hist:直方图均衡化。

# SETTING: Build-in Augmentation
mmseg_random_rotate = 0.1

# SETTING: neural network
lr = 1e-4
iters = 100000 if not debug else 1

# SETTING: Binary Segmentation Mode
SingleChannelMode = True
num_classes = 2
threshold = 0.3 if SingleChannelMode else None
out_channels = 1 if SingleChannelMode else 2
HeadUseSigmoid = True if SingleChannelMode else False
HeadClassWeight = None if SingleChannelMode else [0.1, 1]

# SETTING: Pretrain Task
task_args = {
	'name': 'RelativePositionLearning',
	'samples_per_scan': 1 if debug else 100,
}




# --------------------PARAMETERS-------------------- #


# ////////////////////////////////////////////////// #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# --------------------COMPONENTS-------------------- #






# data preprocess
train_pipeline = [
	dict(type=MMPre_LoadCTImage, task_args=task_args, lmdb_backend_proxy=lmdb_backend_proxy),
	dict(type=Resize, scale=crop_size),
	dict(type=RandomRotate, prob=1, degree=mmseg_random_rotate, pad_val=0, seg_pad_val=0),
	dict(type=RangeClipNorm, 
		 input_shape=crop_size,
		 contrast=contrast, 
		 brightness=brightness, 
		 norm_method=norm_method, 
		 mean=mean, std=std),
	dict(type=PackInputs)
]
test_pipeline = [
	dict(type=MMPre_LoadCTImage, task_args=task_args, lmdb_backend_proxy=lmdb_backend_proxy),
	dict(type=Resize, scale=crop_size),
	dict(type=RangeClipNorm, 
		 input_shape=crop_size,
		 contrast=contrast, 
		 brightness=brightness, 
		 norm_method=norm_method, 
		 mean=mean, std=std),
	dict(type=PackInputs)
]


train_dataset = dict(
	type=RelativePositionLearningDataset, 
	lmdb_backend_proxy=lmdb_backend_proxy if use_lmdb else None,
	task_args=task_args,
	pipeline=train_pipeline,
	database_args=dict(
		root=data_root, 
		metadata_ckpt=meta_ckpt,
		split='train',
	),
	debug=debug,
)
val_dataset = dict(
	type=RelativePositionLearningDataset, 
	lmdb_backend_proxy=lmdb_backend_proxy if use_lmdb else None,
	task_args=task_args,
	pipeline=test_pipeline,
	database_args=dict(
		root=data_root, 
		metadata_ckpt=meta_ckpt,
		split='val',
	),
	debug=debug,
)
test_dataset = dict(
	type=RelativePositionLearningDataset, 
	lmdb_backend_proxy=lmdb_backend_proxy if use_lmdb else None,
	task_args=task_args,
	pipeline=test_pipeline,
	database_args=dict(
		root=data_root, 
		metadata_ckpt=meta_ckpt,
		split='test',
	),
	debug=debug,
)

train_dataloader.update(dataset=train_dataset)
val_dataloader.update(dataset=val_dataset)
test_dataloader.update(dataset=test_dataset)

val_evaluator = dict(type=RelativePositionError, loss_type='L1')
test_evaluator = val_evaluator


model = dict(type=RelativePositionLearning)

# data preprocessor
dpd.update(size=crop_size)
data_preprocessor.update(size=crop_size)

# Optim
optim_wrapper.update(optimizer=dict(lr=lr))
param_scheduler = [
	dict(
		type=LinearLR, start_factor=1e-2, begin=0, end=iters*0.1,
		by_epoch=False),
	dict(
		type=PolyLR,
		eta_min=1e-2*lr,
		power=0.3,
		begin=iters*0.2,
		end=iters,
		by_epoch=False)
]

# Task Control
train_cfg = dict(type=IterBasedTrainLoop, max_iters=iters, val_interval=val_interval)

