import logging
import os
import pdb
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Sequence
from typing import Any
from multiprocessing import Pool, cpu_count
from concurrent import futures

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from mmengine.config import ConfigDict
from mmengine.logging import print_log
from mmengine.evaluator.metric import BaseMetric
from mmcv.transforms import BaseTransform

from mmpretrain.registry import MODELS
from mmpretrain.registry import TRANSFORMS
from mmpretrain.datasets import BaseDataset
from mmpretrain.models.selfsup import BaseSelfSupervisor
from mmpretrain.structures import DataSample

from mmseg.CQK_Med import LoadCTImage
from mmseg.CQK_Med import LMDB_MP_Proxy
from mmseg.CQK_Med import LMDB_DataBackend
from mmseg.CQKGastricCancerCT import MMPreSampleProvider
from mmseg.models import SegformerHead


"""
	CQK_Med_Pretrain_Dataset:	mmseg框架下的dataset规范，主要功能是生成样本索引列表
	MMPreSampleProvider:		数据集索引文件的实现和对外接口，能够按照数据集文件结构选取索引并返回
	LMDB_MP_Proxy:				同时实现 —— Python多进程共享变量、mmseg全局变量。帮助框架在任意位置实现LMDB数据库的并发访问
	LMDB_DataBackend:			数据集的LMDB数据库封装，定义键的命名规则，实现数据集文件结构至数据库结构的映射
	SelfSupLabelGenerator:		自监督算法的标签生成器，需要由lmdb数据库提供的样本元数据
	MMPre_LoadCTImage:			Dataloader Worker Task，实现训练时并行生成标签
	RelativePositionLearning:	定义自监督算法任务流，从属于mmseg.classifier
	RelativePositionError:		逐点误差函数
"""

class CQK_Med_Pretrain_Dataset(BaseDataset):
	def __init__(self, task_args:Dict, lmdb_backend_proxy:Optional[ConfigDict], 
				 database_args:Dict, debug:bool, *args, **kwargs) -> None:
		"""
		Args:
			task_args: Dict for Task Configuration
			database_args: Dict ['root', 'metadata_ckpt', 'split']
		"""
		# 如果使用lmdb数据库，从lmdb数据库中读取索引文件，以字典形式传入CT数据集接口
		if lmdb_backend_proxy:
			assert isinstance(lmdb_backend_proxy, ConfigDict), "[DATASET] lmdb_backend_proxy must be a dict"
			lmdb_backend_proxy:LMDB_MP_Proxy = TRANSFORMS.build(lmdb_backend_proxy)
			self.lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()
			meta_dict = self.lmdb_service.meta_data_dict(database_args['metadata_ckpt'])
			if meta_dict:
				database_args['metadata_ckpt'] = meta_dict
			else:
				FileNotFoundError(f"metadata_ckpt {database_args.get('metadata_ckpt')} not found from lmdb_backend_proxy")

		self._database_args = database_args
		self.CT_Database = MMPreSampleProvider(database_args['root'], database_args['metadata_ckpt'])
		self.task_args = task_args
		self.debug = debug

		super().__init__(ann_file='', *args, **kwargs)

	def _SampleList():
		raise NotImplementedError

	def load_data_list(self) -> List[dict]:
		sample_list = self._SampleList()
		print_log(f"[Dataset] 索引数据集 | split:{self._database_args['split']} | num_sam:{len(sample_list)}", "current", logging.INFO)
		return sample_list[:32] if self.debug else sample_list


class RelativePositionLearningDataset(CQK_Med_Pretrain_Dataset):
	@staticmethod
	def _SampleList_process_one(meta_bytes:bytes, path:str):
		(meta, _) = LMDB_DataBackend.decompress(meta_bytes, None)
		ScanStartLocation = meta.get('00271050').get('Value')[0]	# start - high value
		ScanEndLocation = meta.get('00271051').get('Value')[0]		#  end  - low  value
		ScanLength = ScanStartLocation - ScanEndLocation
		if ScanLength<1: path = None
		return path
	def _SampleList(self):
		assert hasattr(self, 'lmdb_service'), '该学习方法需要lmdb数据库提供Meta Data'
		sample_list = self.CT_Database.MMPRE_RawList(self._database_args['split'], **self.task_args)	
		
		# 筛选掉所有ScanStartLocation和ScanEndLocation均为0的样本
		# map reduce: 读取所有meta的buffer
		mp_pool = Pool(cpu_count())
		with futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
			# 子线程获取bytes，交由进程池对bytes进行分析
			# 进程池返回非阻塞fetcher，保证线程执行不被进程阻塞，也即bytes的获取不被分析过程阻塞
			executions = [executor.submit(self.lmdb_service.fetch_data, sample['img_path'], True, False)
						  for sample in sample_list]
			path_fetcher = []
			for future in tqdm(futures.as_completed(executions), total=len(sample_list),
								desc='[Dataset] 清洗SlicePosition非法样本 读取buffer中'):
				path, meta_bytes, _ = future.result()
				
				fetch = mp_pool.apply_async(self._SampleList_process_one, (meta_bytes, path))
				path_fetcher.append(fetch)	# 如果不合法，就会返回None。正常情况下会返回这个样本的path
		
		valid_sample_list=[]
		for fetcher in path_fetcher:
			path = fetcher.get()
			if path: valid_sample_list.append({'img_path': path})
		
		print_log(f"[Dataset] 从{self._database_args['split']}集中共清除{len(sample_list)-len(valid_sample_list)}个非法样本", 'current', logging.INFO)
		return valid_sample_list


class SelfSupervisorWithValidation(BaseSelfSupervisor):
	def extract_feat(self, inputs: Tensor):
		x = super().extract_feat(inputs)
		if self.with_neck: x = self.neck(x)
		return x

	def forward(self,
				inputs: Union[torch.Tensor, List[torch.Tensor]],
				data_samples: Optional[List[DataSample]] = None,
				mode: str = 'tensor'):
		if mode == 'tensor':
			return self.extract_feat(inputs)
		elif mode == 'loss':
			return self.loss(inputs, data_samples)
		elif mode == 'predict':
			return self.predict(inputs, data_samples)
		else:
			raise RuntimeError(f'Invalid mode "{mode}".')


# 预训练任务流：切片位置学习
class RelativePositionLearning(SelfSupervisorWithValidation):
	def loss(self, inputs: torch.Tensor,
			 data_samples: List[DataSample]) -> dict:
		targets = [sample.gt_label for sample in data_samples]
		targets = torch.concat(targets, dim=0).unsqueeze(-1)    # (N, 1)
		feats = self.extract_feat(inputs)
		return self.head.loss(feats, targets)
	
	def predict(self, 
				inputs: Tensor, 
				data_samples: List[DataSample] | None = None
				) -> List[DataSample]:
		feats = self.extract_feat(inputs)
		preds: torch.Tensor = self.head.predict(feats)
		for i, sample in enumerate(data_samples):
			sample.set_pred_label(preds[i])
		return data_samples



class RelativePositionError(BaseMetric):
	def __init__(self, loss_type:str='L1', *args, **kwargs):
		assert loss_type in ['L1', 'L2']
		super().__init__(prefix='Dist', *args, **kwargs)
		self.results: List[torch.Tensor] = []
		self.loss_type = loss_type
		self.loss_fc = F.l1_loss if loss_type == 'L1' else F.mse_loss
	
	def process(self, data_batch, data_samples: Sequence[dict]):
		for data_sample in data_samples:
			loss = self.loss_fc(data_sample['pred_label'].cpu(), data_sample['gt_label'].float().cpu())
			self.results.append(loss)
	
	def compute_metrics(self, results: list) -> dict:
		return {f"RelPos_{self.loss_type}": torch.stack(results).mean().cpu()}



class MMPre_LoadCTImage(LoadCTImage):
	def __init__(self, task_args: Dict, lmdb_backend_proxy: ConfigDict = None) -> None:
		# 初始化数据库后端，方法与mmseg相同
		super().__init__(lmdb_backend_proxy)
		# 预训练或自监督的各种任务参数设定。在进行前向时，实时生成标签。
		self.label_generater = SelfSupLabelGenerator(task_args)

	def transform(self, results: Dict) -> Dict:
		# 与mmseg一样，正常加载图像，这是每个任务都需要做的基本步骤
		results = super().transform(results)	# keys: ['img_path', 'dcm_meta', 'img', 'img_shape', 'ori_shape']
		
		# 生成独属于预训练的label
		(label, valid) = self.label_generater(results['dcm_meta'], results['img'])
		results['gt_label'] = label
		# 若该样本无效，置零图像
		if not valid:
			results['img'] = np.zeros_like(results['img'])
		
		return results
		

# 预训练标签生成器，根据不同的任务返回不同的标签。
# 标签实时生成，集成在Dataloader worker工作流中
class SelfSupLabelGenerator:
	def __init__(self, task_args):
		self.task_args = task_args

	def RelativePositionLearning(self, meta) -> float:
		ScanStartLocation = meta.get('00271050').get('Value')[0]	# start - high value
		SliceLocation = meta.get('00201041').get('Value')[0]		# slice - mid  value
		ScanEndLocation = meta.get('00271051').get('Value')[0]		#  end  - low  value
		ScanLength = ScanStartLocation - ScanEndLocation
		if ScanLength==0: 
			print(f"Invalid ScanLocation in CT image: {meta.get('00100010').get('Value')}")
			return (0, False)

		# From Start to End: [0, length]
		SliceLocationPercent = (ScanStartLocation - SliceLocation) / ((SliceLocation - ScanEndLocation)+1)
		# Scale
		label = SliceLocationPercent / 10
		return (label, True)

	def LocationLearning(self, meta) -> Tuple[float, float, float]:
		raise NotImplementedError	# [x,y,z]

	def __call__(self, meta, img) -> Tuple[Any, bool]:
		if self.task_args['name'] == 'RelativePositionLearning':
			return self.RelativePositionLearning(meta)



class SegformerNeck(SegformerHead):
	def forward(self, x):
		return (super().forward(x), )
