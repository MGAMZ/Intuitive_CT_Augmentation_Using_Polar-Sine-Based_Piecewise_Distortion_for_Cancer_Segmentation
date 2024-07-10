# mmsegmentation
from mgamdata.CTGastricCancer2023 import GastricCancer_2023
from mgamdata.CTGastricCancer2023 import LMDB_MP_Proxy
from mgamdata.CTGastricCancer2023 import LoadCTImage
from mgamdata.CTGastricCancer2023 import LoadCTLabel
from mgamdata.CTGastricCancer2023 import CTSegVisualizationHook
from mgamdata.DistortionAugment import ScanTableRemover
from mgamdata.DistortionAugment import Distortion
from mgamdata.DistortionAugment import RangeClipNorm
from mmseg.datasets.transforms import PackSegInputs
from mmseg.datasets.transforms import Resize
from mmseg.datasets.transforms import RandomRotate

from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.visualization import SegLocalVisualizer
from mmseg.evaluation.metrics import IoUMetric

# mmengine
from mmengine.runner import IterBasedTrainLoop
from mmengine.runner import ValLoop
from mmengine.runner import TestLoop
from mmengine.dataset import InfiniteSampler1
from mmengine.dataset import DefaultSampler
from mmengine.optim import AmpOptimWrapper
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import LinearLR
from mmengine.optim.scheduler import PolyLR
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.logger_hook import LoggerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks import DistSamplerSeedHook
from mmengine.visualization import LocalVisBackend
from mmengine.visualization import TensorboardVisBackend
from mmengine.dataset.sampler import InfiniteSampler

# Neural Network
from torch.optim import AdamW
from mmseg.models import EncoderDecoder
from mmseg.models import CrossEntropyLoss
from mmseg.models import DiceLoss
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.normalization import GroupNorm



debug=False
use_AMP = True
compile = True if not debug else False

data_root = '/mnt/e/mgam_ct/2023_Med_CQK'
meta_ckpt = '/mnt/e/mgam_ct/2023_Med_CQK/V3_2024-03-16_14-59-01.pickle'
use_lmdb = True
lmdb_backend_proxy = {
	"dataset_root": data_root,
	"lmdb_path": r"/mnt/e/mgam_ct/2023_Med_CQK/lmdb_database",
	"mode": "normal",
} if use_lmdb else None
batch_size = 8
workers = 0 if debug else 8
data_preprocessor_normalize=False
mean, std = 561.54, 486.59              # before clip to (0,4095): (10.87, 1138.37), after: (561.54, 486.59)
source_img_shape = (512,512)
crop_size = (256,256)
val_interval = 1000 if not debug else 1


train_dataloader = dict(
	num_workers = workers,
	batch_size=batch_size,
	pin_memory=True,
	persistent_workers=True if workers > 0 else False,
	prefetch_factor= 4 if workers > 0 else None,
	sampler=dict(type=InfiniteSampler, shuffle=True),
)
val_dataloader = dict(
	num_workers = workers//2,
	batch_size=batch_size,
	pin_memory=False,
	persistent_workers=True if workers > 0 else False,
	prefetch_factor= 4 if workers > 0 else None,
	sampler=dict(type=DefaultSampler, shuffle=False),
)
test_dataloader = dict(
	num_workers = workers//2,
	batch_size=batch_size,
	pin_memory=False,
	persistent_workers=True if workers > 0 else False,
	prefetch_factor= 4 if workers > 0 else None,
	sampler=dict(type=DefaultSampler, shuffle=False),
)


dpd = data_preprocessor = dict(
	type=SegDataPreProcessor,
	mean=mean if data_preprocessor_normalize else None,
	std=std if data_preprocessor_normalize else None,
	pad_val=0,
	size=crop_size,
	seg_pad_val=0,
	non_blocking=True,
)


model = dict(
	type=EncoderDecoder,
	data_preprocessor=dpd,
	train_cfg=dict(),
	test_cfg=dict(mode='whole'),
)

# evaluation
val_evaluator = test_evaluator = dict(type=IoUMetric, iou_metrics=['mIoU','mDice','mFscore'], nan_to_num=0)

# optimizer and scheduler
optim_wrapper = dict(
	type=AmpOptimWrapper if use_AMP else OptimWrapper,
	optimizer=dict(type=AdamW),
	clip_grad=dict(max_norm=1, 
				   norm_type=2, 
				   error_if_nonfinite=False)
)

# Task Control
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
default_hooks = dict(
	timer=dict(type=IterTimerHook),
	logger=dict(type=LoggerHook, interval=100, log_metric_by_epoch=False),
	param_scheduler=dict(type=ParamSchedulerHook),
	checkpoint=dict(type=CheckpointHook, by_epoch=False),
	sampler_seed=dict(type=DistSamplerSeedHook),
	visualization=dict(type=CTSegVisualizationHook,
					   reverse_stretch=None, draw=True, interval=1 if debug else 5, 
					   lmdb_backend_proxy=lmdb_backend_proxy if use_lmdb else None)
)

# runtime env
runner_type='mgam_Runner'
env_cfg = dict(
	mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
	dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type=LocalVisBackend), dict(type=TensorboardVisBackend)]
visualizer = dict(type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer', alpha=0.4)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
work_dir = './work_dirs/'
tta_model = None
