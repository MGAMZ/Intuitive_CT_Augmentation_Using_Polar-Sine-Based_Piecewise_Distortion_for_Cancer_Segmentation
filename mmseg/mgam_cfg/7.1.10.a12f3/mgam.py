from mmengine.config import read_base
with read_base():
	from ..mgam_base import *



# SETTING: sample augment
num_positive_img = 1                    # 每个病例正样本采样数
num_negative_img = 0                    # 每个病例负样本采样数
minimum_negative_distance = 50          # 正负样本交界面与标注面的距离

# SETTING: Normalization
contrast = 1                            # 全局对比度控制，影响标准差
brightness = 0                          # 全局亮度控制，影响均值
norm_method = 'const'                   # const:按照训练集分布归一化, inst:实例归一化。仅针对CTImageEnhance方法。hist:直方图均衡化。
HistEqual=True if norm_method=='hist' else False        # 是否进行直方图均衡化
HistEqualUseMask=True                   # 进行直方图均衡化是否基于mask
if norm_method == 'inst': assert data_preprocessor_normalize == False, \
	"Instance Norm 只能用CTImageEnhance实现，此时mmseg自带的预处理器应当禁用归一化"

# SETTING: Distortion
distort_global_rotate=0                 # 全局旋转角度，随机介于[-90x°, 90x°]
distort_amplitude=12                    # 扭曲的振幅，沿中心向四周仿射的正弦振幅
distort_frequency=3                     # 扭曲的频率，沿中心向四周仿射的正弦频率
distort_grid_dense=15                   # 分段仿射映射的网格密度
distort_refresh_interval=20             # 每隔多少个扭曲调用刷新一次映射矩阵

# SETTING: TableRemove
TableRemove_IntrinsicCircleRadius = 600
TableRemove_MaskOffset = 0

# SETTING: Build-in Augmentation
mmseg_random_rotate = 0.01

# SETTING: neural network

lr = 1e-4
iters = 3000 if not debug else 1

# SETTING: Binary Segmentation Mode
SingleChannelMode = True
num_classes = 2
threshold = 0.3 if SingleChannelMode else None
out_channels = 1 if SingleChannelMode else 2
HeadUseSigmoid = True if SingleChannelMode else False
HeadClassWeight = None if SingleChannelMode else [0.1, 1]




# --------------------PARAMETERS-------------------- #


# ////////////////////////////////////////////////// #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


# --------------------CONPONENTS-------------------- #



# data preprocess
train_pipeline = [
	dict(type=LoadCTImage, lmdb_backend_proxy=lmdb_backend_proxy),
	# dict(type=ScanTableRemover, 
	# 	 TableIntrinsicCircleRadius=TableRemove_IntrinsicCircleRadius, 
	# 	 MaskOffset=TableRemove_MaskOffset, 
	# 	 pixel_array_shape=source_img_shape),
	dict(type=LoadCTLabel, lmdb_backend_proxy=lmdb_backend_proxy),
	dict(type=Resize, scale=crop_size),
	# dict(type=RandomRotate, prob=1, degree=mmseg_random_rotate, pad_val=0, seg_pad_val=0),
	dict(type=Distortion,
		 global_rotate=distort_global_rotate,
		 amplitude=distort_amplitude,
		 frequency=distort_frequency,
		 grid_dense=distort_grid_dense,
		 in_array_shape=crop_size,
		 refresh_interval=distort_refresh_interval),
	dict(type=RangeClipNorm, 
		 input_shape=crop_size,
		 contrast=contrast, 
		 brightness=brightness, 
		 norm_method=norm_method, 
		 mean=mean, std=std),
	dict(type=PackSegInputs)
]
test_pipeline = [
	dict(type=LoadCTImage, lmdb_backend_proxy=lmdb_backend_proxy),
	# dict(type=ScanTableRemover, 
	# 	 TableIntrinsicCircleRadius=TableRemove_IntrinsicCircleRadius, 
	# 	 MaskOffset=TableRemove_MaskOffset, 
	# 	 pixel_array_shape=source_img_shape),
	dict(type=Resize, scale=crop_size),
	dict(type=RangeClipNorm, 
		 input_shape=crop_size,
		 contrast=contrast, 
		 brightness=brightness, 
		 norm_method=norm_method, 
		 mean=mean, std=std),
	dict(type=LoadCTLabel, lmdb_backend_proxy=lmdb_backend_proxy),
	dict(type=PackSegInputs)
]




train_dataset = dict(
	type=CQK_2023_Med, 
	pipeline=train_pipeline,
	database_args=dict(
		root=data_root, 
		metadata_ckpt=meta_ckpt,
		split='train',
		num_positive_img=num_positive_img,
		num_negative_img=num_negative_img,
		minimum_negative_distance=minimum_negative_distance,
	),
	debug=debug,
)
val_dataset = dict(
	type=CQK_2023_Med, 
	pipeline=test_pipeline,
	database_args=dict(
		root=data_root, 
		metadata_ckpt=meta_ckpt,
		split='val',
		num_positive_img=num_positive_img,
		num_negative_img=num_negative_img,
		minimum_negative_distance=minimum_negative_distance,
	),
	debug=debug,
)
test_dataset = dict(
	type=CQK_2023_Med, 
	pipeline=test_pipeline,
	database_args=dict(
		root=data_root, 
		metadata_ckpt=meta_ckpt,
		split='test',
		num_positive_img=num_positive_img,
		num_negative_img=num_negative_img,
		minimum_negative_distance=minimum_negative_distance,
	),
	debug=debug,
)

train_dataloader.update(dataset=train_dataset)
val_dataloader.update(dataset=val_dataset)
test_dataloader.update(dataset=test_dataset)

# data preprocessor
dpd.update(size=crop_size)
data_preprocessor.update(size=crop_size)

# train
train_cfg = dict(type=IterBasedTrainLoop, max_iters=iters, val_interval=val_interval)

# Optim
optim_wrapper.update(optimizer=dict(lr=lr))
param_scheduler = [
	dict(
		type=LinearLR, start_factor=1e-1, begin=0, end=iters*0.1,
		by_epoch=False),
	dict(
		type=PolyLR,
		eta_min=1e-1*lr,
		power=0.3,
		begin=iters*0.7,
		end=iters,
		by_epoch=False)
]





