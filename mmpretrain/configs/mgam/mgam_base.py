# mmsegmentation
from mmseg.CQK_Med import LMDB_MP_Proxy
from mmseg.CQK_Med import ScanTableRemover
from mmseg.CQK_Med import RangeClipNorm
from mmseg.datasets.transforms import Resize
from mmseg.datasets.transforms import RandomRotate
from mmseg.evaluation.metrics import IoUMetric

# mmpretrain
from mmpretrain.datasets.transforms import PackInputs
from mmpretrain.visualization import UniversalVisualizer
from mmpretrain.engine.hooks import VisualizationHook
from mmpretrain.models.utils.data_preprocessor import SelfSupDataPreprocessor

# mmengine
from mmengine.runner import ValLoop
from mmengine.runner import TestLoop
from mmengine.dataset import InfiniteSampler
from mmengine.dataset import DefaultSampler
from mmengine.optim import AmpOptimWrapper
from mmengine.optim import OptimWrapper
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.logger_hook import LoggerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks import DistSamplerSeedHook
from mmengine.visualization import LocalVisBackend
from mmengine.visualization import TensorboardVisBackend
from mmengine.optim.scheduler import LinearLR
from mmengine.optim.scheduler import PolyLR
from mmengine.runner import IterBasedTrainLoop

# Neural Network
from torch.optim import AdamW
from mmseg.models import EncoderDecoder
from mmseg.models import CrossEntropyLoss
from mmseg.models import DiceLoss
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.normalization import GroupNorm

debug=False

use_AMP = True
compile = False		# 启动时间很慢，还不知为什么，可能需要compile模型

data_root = r"../2023_Med_CQK"
meta_ckpt = data_root + '/V3_2024-03-16_14-59-01.pickle'
use_lmdb = True
lmdb_backend_param = {
	"dataset_root": data_root,
	"lmdb_path": data_root + '/lmdb_database',
	"mode": "normal",
}
batch_size = 4 if debug else 8
workers = 0 if debug else 8
mean, std = 561.54, 486.59              # before clip to (0,4095): (10.87, 1138.37), after: (561.54, 486.59)
source_img_shape = (512,512)
crop_size = (256,256)
val_interval = 1000 if not debug else 1


lmdb_backend_proxy = dict(
	type=LMDB_MP_Proxy,
	name="LMDB_MP_Proxy",
	lmdb_args=lmdb_backend_param,
) if use_lmdb else None


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
	type=SelfSupDataPreprocessor,
	mean=None,
	std=None,
	pad_val=0,
	size=crop_size,
	non_blocking=True,
)

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
	logger=dict(type=LoggerHook, interval=1 if debug else 100, log_metric_by_epoch=False),
	param_scheduler=dict(type=ParamSchedulerHook),
	checkpoint=dict(type=CheckpointHook, by_epoch=False, save_last=True, max_keep_ckpts=1, published_keys='backbone.'),
	sampler_seed=dict(type=DistSamplerSeedHook),
	visualization=dict(type=VisualizationHook, enable=False)
)

# runtime env
runner_type='mgam_Runner'
env_cfg = dict(
    cudnn_benchmark=True,
    cuda_matmul_allow_tf32=True,
    cudnn_allow_tf32=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type=LocalVisBackend), dict(type=TensorboardVisBackend)]
visualizer = dict(type=UniversalVisualizer, vis_backends=vis_backends)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
work_dir = './work_dirs/'
randomness = dict(seed=None, deterministic=False)
