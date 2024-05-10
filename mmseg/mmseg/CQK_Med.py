import math
import logging
import random
import pickle
import lmdb
import os
import pdb
import os.path as osp
from io import BytesIO
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from multiprocessing.managers import BaseManager
from pprint import pprint
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Any
from typing import Callable
from typing import Sequence
from lzma import compress
from lzma import decompress
from lzma import FORMAT_XZ

import torch
import pydicom
import nrrd
import numpy as np
from tqdm import tqdm
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import warp
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_hist

from mmseg.registry import DATASETS
from mmseg.registry import TRANSFORMS
from mmseg.registry import HOOKS
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.CQKGastricCancerCT import CQKGastricCancerCT
from mmseg.structures import SegDataSample
from mmseg.engine.hooks import SegVisualizationHook
from mmengine.runner import Runner

from mmengine.utils import ManagerMixin
from mmengine.logging import print_log
from mmengine import ConfigDict
from mmcv.transforms.base import BaseTransform












def rectangular_to_polar(x, y, center_x, center_y):
	"""
	直角坐标由0开始计数
	标准直角坐标系输入: x,y
	极点的直角坐标: center_x, center_y
	
	radius: 极径
	angle: 极角 弧度制
	"""
	# 使用numpy计算半径
	radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
	# 使用scipy计算角度
	angle = np.arctan2(y - center_y, x - center_x)
	
	return radius, angle



def polar_to_rectangular(radius, angle, center_x, center_y):
	"""
	直角坐标由0开始计数
	radius: 极径
	angle: 极角 弧度制
	center_x, center_y: 极点的直角坐标

	x,y: 直角坐标
	"""
	x = center_x + radius * math.cos(angle)
	y = center_y + radius * math.sin(angle)
	
	return x, y


@DATASETS.register_module()
class CQK_2023_Med(BaseSegDataset):
	METAINFO = dict(
		classes=('normal','cancer'),
		palette=[[0], [255]],
	)

	def __init__(self, database_args:dict, debug:bool=False, *args, **kwargs):
		if database_args.get("lmdb_backend_proxy", None):
			assert isinstance(database_args["lmdb_backend_proxy"], ConfigDict), "[DATASET] lmdb_backend_proxy must be a dict"
			lmdb_backend_proxy:LMDB_MP_Proxy = TRANSFORMS.build(database_args["lmdb_backend_proxy"])
			lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()
			meta_key_name = "REGISTRY_"+os.path.basename(database_args["metadata_ckpt"])
			meta_dict = lmdb_service.get(meta_key_name)
			if meta_dict:
				database_args["metadata_ckpt"] = meta_dict
			else:
				FileNotFoundError(f"metadata_ckpt {meta_key_name} not found from lmdb_backend_proxy")

		self.debug = debug
		self._database_args = database_args
		self._DATABASE = CQKGastricCancerCT(**database_args)

		super().__init__(*args, **kwargs)
	
	def load_data_list(self):
		print_log(f"[Dataset] 索引数据集 | split:{self._database_args['split']} | p:{self._database_args['num_positive_img']} | n:{self._database_args['num_negative_img']} | d:{self._database_args['minimum_negative_distance']}", 
				  "current", logging.INFO)
		data_list = self._DATABASE.MMSEG_Segmentation_PosNegEnhance(
						self._database_args['split'],
						self._database_args['num_positive_img'],
						self._database_args['num_negative_img'],
						self._database_args['minimum_negative_distance']
					)	# list[dict]
		print_log(f"[Dataset] 数据集索引完成 | split:{self._database_args['split']} | num_sam:{len(data_list)}", 
				  "current", logging.INFO)
		return data_list[:32] if self.debug else data_list



class CTSegDatasetWithDistortionCache(CQK_2023_Med):
	def __init__(self, distortion_args:dict, *args, **kwgs):
		super().__init__(*args, **kwgs)
		self.distorter = Distortion(**distortion_args)

	def _pregenerate_distortion(self, data_list:List[str]):
		return self.distorter.multiprocess_distort(data_list)

	def load_data_list(self):
		data_list = super().load_data_list()
		self.data_cache = self._pregenerate_distortion(data_list)
		return self.data_cache



class moving_average_filter(object):
	def __init__(self, window_size:int=100) -> None:
		self.window_size = window_size
		self.data = []
	
	def __call__(self, data:float) -> float:
		self.data.append(data)
		if len(self.data) > self.window_size:
			self.data.pop(0)
		return sum(self.data) / len(self.data)



class LMDB_DataBackend:
	COMPRESS_FORMAT = FORMAT_XZ				# 压缩格式
	COMPRESS_PRESET = 5						# 压缩等级
	MAX_MAP_SIZE = 1024 * 1024 * 1024 * 72	# LMDB容量：72G
	PIXEL_ARRAY_TYPE = np.int16				# 图像像素格式

	def __init__(self, dataset_root:str, lmdb_path:str, mode:str=None) -> Any:
		self.dataset_root = dataset_root
		self.lmdb_path = lmdb_path
		print(f"正在加载LMDB数据库{lmdb_path}")
		# 检查该文件是否存在
		if mode=="init":
			self._init_lmdb()
		elif mode=="normal" or mode=='check':
			self.lmdb_env = lmdb.Environment(self.lmdb_path, LMDB_DataBackend.MAX_MAP_SIZE, 
											 readonly=True, create=False, lock=False, readahead=False)
		elif mode=="debug":
			self.lmdb_env = lmdb.Environment(self.lmdb_path, LMDB_DataBackend.MAX_MAP_SIZE, 
											 readonly=False, create=False)
		else:
			raise NotImplementedError("mode must be in ['init', 'normal']")
		print(f"LMDB数据库env已建立")

		if mode=='check':
			self._check_all_items()


	@classmethod
	def _check_one_item(cls, key, value) -> Dict[str, Any]:
		try:
			if key.decode().startswith('METADATA'):
				decompressed_meta = cls._decompress_meta(value)
				if not isinstance(decompressed_meta, Dict):
					return {'Error Type':'Invalid Decompressed Meta',
							'key': key.decode(),
							'decompressed': decompressed_meta}
			elif key.decode().startswith('PIXEL_ARRAY'):
				decompressed_pixel = cls._decompress_meta(value)
				if not isinstance(decompressed_pixel, Dict):
					return {'Error Type':'Invalid Decompressed Pixel',
							'key': key.decode(),
							'decompressed': decompressed_pixel}
			else:
				return {'Error Type':'Invalid Key',
						'key': key.decode(),
						'decompressed': None}
			
			return None
			
		except Exception as e:
			return {'Error Type': e,
					'key': key.decode(),
					'decompressed': None}

	def _check_all_items(self):
		txn = self.lmdb_env.begin(write=False)
		cursor = txn.cursor()
		results = []
		failed_count = 0
		logger = logging.getLogger('lmdb_check')
		logging.basicConfig(filename='lmdb_check.log', filemode='w', level=logging.DEBUG)
		console_handler = logging.StreamHandler()
		logger.addHandler(console_handler)
		
		with Pool(cpu_count()) as p:
			pbar = tqdm(cursor, desc='部署任务')
			for key, value in pbar:
				results.append(p.apply_async(self._check_one_item, args=(key, value)))
			pbar = tqdm(results, desc='收集结果', total=len(results))
			for info in pbar:
				info = info.get()
				if info is not None: 
					pprint(info)
					logger.debug(info)
					failed_count += 1
			
		logger.info(f"{failed_count} failed")

	@classmethod
	def _init_lmdb_process_one_file(cls, params) -> Dict[Union[str,bytes], Union[bytes,float]]:
		dataset_root, file_root, file = params
		source_type = file.split(".")[-1]
		source_path = os.path.join(file_root, file)
		npy_path = source_path.replace(".dcm", ".npy").replace(".nrrd", ".npy")
		
		metadata_key_name, pixel_array_key_name = LMDB_DataBackend._map_path_key(dataset_root, source_path)

		# MetaData from dcm
		meta_buffer = BytesIO()
		if source_type == "dcm":
			dcm_data = pydicom.dcmread(source_path)
			meta = dcm_data.to_json_dict()
			meta.pop('7FE00010')	# 移除dcm中的像素数据，这部分已经由npy提供了
		elif source_type == "nrrd":
			pixel, meta = nrrd.read(source_path)
		else:
			raise NotImplementedError(f"source_type only support ['dcm', 'nrrd'], but got {source_type}")
		meta:Dict
		pickle.dump(meta, meta_buffer)

		# Pixel from npy
		npy_buffer = BytesIO()
		npy_data = np.load(npy_path).astype(cls.PIXEL_ARRAY_TYPE)
		np.save(npy_buffer, npy_data, allow_pickle=False)

		# lzma压缩
		compressed_meta_byte = compress(meta_buffer.getvalue(), format=cls.COMPRESS_FORMAT, preset=cls.COMPRESS_PRESET)
		compressed_pixel_byte = compress(npy_buffer.getvalue(), format=cls.COMPRESS_FORMAT, preset=cls.COMPRESS_PRESET)
		compress_ratio = len(compressed_meta_byte+compressed_pixel_byte)/len(meta_buffer.getvalue()+npy_data.tobytes())

		return {metadata_key_name: compressed_meta_byte, 
				pixel_array_key_name: compressed_pixel_byte,
				"compress_ratio": compress_ratio}

	# 从零构建lmdb数据库
	def _init_lmdb(self):
		self.lmdb_env = lmdb.Environment(self.lmdb_path, LMDB_DataBackend.MAX_MAP_SIZE)
		compress_ratio_filter = moving_average_filter(window_size=100)

		from multiprocessing import Pool, cpu_count
		exec_pool = Pool(cpu_count())
		task_params = []
		txn = self.lmdb_env.begin(write=False)

		# 获取生成长度
		walk_step = 0
		for _ in os.walk(self.dataset_root):
			walk_step += 1
		pbar1 = tqdm(os.walk(self.dataset_root), desc="检查lmdb数据库完整性", total=walk_step)
		for roots, dirs, files in pbar1:
			for file in files:
				# 由于先前的数据处理中，数据集已经经过一次清洗和转换
				# 现在每一个dcm和nrrd文件都有一个对应名称的npy文件
				# 因此输出dcm和nrrd的名称，程序能够自动在目录下找到npy文件并读取进数据库
				if file.endswith(".dcm") or file.endswith(".nrrd"):	# 源序列读取
					name = self._map_path_key(self.dataset_root, os.path.join(roots, file))
					if not (txn.get(name[0], False) and txn.get(name[1], False)):
						task_params.append((self.dataset_root, roots, file))
		print(f"总共有{len(task_params)}个文件需要加入lmdb")
		
		# lmdb写入
		fetcher = exec_pool.imap_unordered(self._init_lmdb_process_one_file, task_params)
		pbar = tqdm(fetcher, total=len(task_params), desc="写入源序列")
		for bytes_dict in pbar:
			for key, value in bytes_dict.items():
				if key=="compress_ratio": 
					pbar.set_description(f"写入lmdb | 压缩率: {compress_ratio_filter(float(value))*100:.1f}%")
				else:
					with self.lmdb_env.begin(write=True) as txn:
						txn.put(key, value)
	
	@staticmethod
	def _map_path_key(dataset_root, path:str)->Tuple[bytes, bytes]:
		path = path.split(os.path.basename(dataset_root))[-1]
		path = os.path.join(*path.replace("/","\\").split(".")[:-1])
		metadata_key_name = ("METADATA_" + path).encode()
		pixel_array_key_name = ("PIXEL_ARRAY_" + path).encode()
		return (metadata_key_name, pixel_array_key_name)


	def put(self, key:str, value:bytes):
		if not isinstance(value, bytes):
			raise TypeError("values put into lmdb must be bytes")
		
		compressed = compress(value, self.COMPRESS_FORMAT)
		with self.lmdb_env.begin(write=True) as txn:
			txn.put(key.encode(), compressed)
		
		print(f"WRITE TO LMDB database: {self.lmdb_env.path()}, key:{key}, ori_size: {len(value)/1024:.2f} KB, cmp_size: {len(compressed)/1024:.2f} KB")
		return True


	def get(self, key:str) -> bytes:
		with self.lmdb_env.begin(write=False) as txn:
			compressed = txn.get(key.encode(), None)
			if compressed is None:
				print(f"GET from LMDB database: {self.lmdb_env.path()}, key:{key}, KEY NOT FOUND!!! ")
				return None
			decompressed = decompress(compressed, self.COMPRESS_FORMAT)
			print(f"GET from LMDB database: {self.lmdb_env.path()}, key:{key}, ori_size: {len(decompressed)/1024:.2f} KB cmp_size: {len(compressed)/1024:.2f} KB")
			return decompressed


	def meta_data_dict(self, meta_path:str) -> Dict:
		with self.lmdb_env.begin(write=False) as txn:
			meta_dict_key = 'REGISTRY_'+os.path.basename(meta_path)
			meta_buffer = txn.get(meta_dict_key.encode(), None)
			if meta_buffer is None:
				raise RuntimeError(f"GET from LMDB database: {self.lmdb_env.path()}, key:{meta_dict_key}, KEY NOT FOUND!!! ")
			else:
				return pickle.loads(decompress(meta_buffer, self.COMPRESS_FORMAT))


	def database_test(self):
		# 打印lmdb数据库有关信息
		with self.lmdb_env.begin(write=False) as txn:
			print(f"LMDB数据库信息:{txn.stat()}")
			# 遍历数据库
			for key, value in txn.cursor():
				print(f"key:{key}")
				break
		
		meta, pixel = self(r"./img/柏家荣/ImageFileName000.dcm")
		for key, value in meta.items():
			print(f"meta 键:{key} | 值:{meta.get(key)}")
		print(f"\npixel:{pixel}\n")


	@classmethod
	def _decompress_meta(cls, meta_buffer:bytes) -> Dict | None:
		if meta_buffer is not None:
			meta_buffer = decompress(meta_buffer, format=cls.COMPRESS_FORMAT)
			return pickle.load(BytesIO(meta_buffer))
		else:
			return None

	@classmethod
	def _decompress_pixel(cls, pixel_buffer:bytes) -> np.ndarray | None:
		if pixel_buffer is not None:
			pixel_buffer = decompress(pixel_buffer, format=cls.COMPRESS_FORMAT)
			return np.load(BytesIO(pixel_buffer)).astype(cls.PIXEL_ARRAY_TYPE)
		else:
			return None

	@classmethod
	def decompress(cls, meta_buffer:bytes=None, pixel_buffer:bytes=None) -> Tuple[Union[Dict,None], Union[np.ndarray, None]]:
		meta = cls._decompress_meta(meta_buffer)
		pixel = cls._decompress_pixel(pixel_buffer)
		return (meta, pixel)


	def fetch_data(self, path:str, meta:bool=True, pixel:bool=True) -> Tuple[str, bytes, bytes]:
		meta_key, pixel_key = self._map_path_key(self.dataset_root, path)
		meta_buffer, pixel_buffer = None, None
		with self.lmdb_env.begin(write=False) as txn:
			if meta:
				meta_buffer = txn.get(meta_key, None)
			if pixel:
				pixel_buffer = txn.get(pixel_key, None)
		return (path, meta_buffer, pixel_buffer)



class LMDB_DataBackend_MP_Manager(BaseManager):
	# 无需进行任何定义，但必须继承形成一个新的类
	pass


# 代理执行器
@TRANSFORMS.register_module()
class LMDB_MP_Proxy(ManagerMixin):
	def __init__(self, name:str, lmdb_args:dict) -> None:
		super().__init__(name=name)

		self.lmdb_args = lmdb_args
		LMDB_DataBackend_MP_Manager.register('LMDB_DataBackend', LMDB_DataBackend)
		manager = LMDB_DataBackend_MP_Manager()
		manager.start()
		# 获取一个服务代理，这是一个共享对象
		self.lmdb_service = manager.LMDB_DataBackend(**lmdb_args)


	def transform(self) -> Callable:
		raise NotImplementedError


	def __call__(self) -> LMDB_DataBackend:
		return self.lmdb_service


@TRANSFORMS.register_module()
class LoadCTImage(BaseTransform):
	def __init__(self, lmdb_backend_proxy:ConfigDict=None) -> None:
		super().__init__()
		if lmdb_backend_proxy:
			assert isinstance(lmdb_backend_proxy, ConfigDict), "lmdb_backend_proxy must be a ConfigDict"
			lmdb_backend_proxy:LMDB_MP_Proxy = TRANSFORMS.build(lmdb_backend_proxy)
			self.lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()

	def transform(self, results: Dict) -> Dict:
		if isinstance(results['img_path'], np.ndarray):
			img = results['img_path']
			if hasattr(self, 'lmdb_service'):
				(path, meta_buffer, img_buffer) = self.lmdb_service.fetch_data(results['img_path'])
				(meta, _) = LMDB_DataBackend.decompress(meta_buffer, None)
				results['dcm_meta'] = meta
		else:
			if hasattr(self, 'lmdb_service'):
				(path, meta_buffer, img_buffer) = self.lmdb_service.fetch_data(results['img_path'])
				(meta, img) = LMDB_DataBackend.decompress(meta_buffer, img_buffer)
				results['dcm_meta'] = meta
			else:
				img = np.load(results['img_path'])

		results['img'] = np.clip(img, 0, 4095).astype(np.uint16).squeeze()
		results['img_shape'] = img.shape
		results['ori_shape'] = img.shape
		return results	# img: [H,W,C]


@TRANSFORMS.register_module()
class LoadCTLabel(BaseTransform):
	def __init__(self, lmdb_backend_proxy:ConfigDict=None) -> None:
		super().__init__()
		if lmdb_backend_proxy:
			assert isinstance(lmdb_backend_proxy, ConfigDict), "lmdb_backend_proxy must be a ConfigDict"
			lmdb_backend_proxy:LMDB_MP_Proxy = TRANSFORMS.build(lmdb_backend_proxy)
			self.lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()
			self.EmptyLabel = np.zeros((512,512), dtype=np.uint8)

	def transform(self, results: Dict) -> Dict:
		if results['seg_map_path'] is None:
			gt_seg_map = self.EmptyLabel
		else:
			if hasattr(self, 'lmdb_service'):
				(path, meta_buffer, img_buffer) = self.lmdb_service.fetch_data(results['seg_map_path'])
				(meta, gt_seg_map) = LMDB_DataBackend.decompress(meta_buffer, img_buffer)
			else:
				gt_seg_map = np.load(results['seg_map_path'])
		
		results['gt_seg_map'] = gt_seg_map.astype(np.uint8).squeeze()
		results['seg_fields'].append('gt_seg_map')
		return results	# gt_seg_map: [H,W]


@TRANSFORMS.register_module()
class ScanTableRemover(BaseTransform):
	def __init__(self, pad_val=0, TableIntrinsicCircleRadius=600, MaskOffset=-12, pixel_array_shape=(512,512)):
		if len(pixel_array_shape) == 1:
			self.pixel_array_shape = pixel_array_shape
		elif len(pixel_array_shape) == 2:
			assert pixel_array_shape[0] == pixel_array_shape[1], "pixel_array_shape should be a square"
			self.pixel_array_shape = pixel_array_shape[0]
		else:
			raise ValueError("Only 2D array is supported")
		self.pad_val = pad_val
		self.TableIntrinsicCircleRadius = TableIntrinsicCircleRadius
		self.MaskOffset = MaskOffset
		self.PixelSpacing_offset = 1

	def process(self, pixel_array:np.ndarray, table_height:float, 
				recon_center_cord:Tuple[float,float], 
				pixel_spacing:float):
		if len(pixel_spacing) == 2:
			if pixel_spacing[0] == pixel_spacing[1]:
				pixel_spacing = pixel_spacing[0]
			else:
				raise ValueError("If 2D array, only square shape is supported")
		pixel_spacing = pixel_spacing * self.PixelSpacing_offset

		# Slice中央垂直各要素距离计算
		Position_MaskCenter = table_height + self.MaskOffset - self.TableIntrinsicCircleRadius
		Distance_MaskCenter_ReconCenter = recon_center_cord[0] - Position_MaskCenter

		# 物理单位(mm) -> 图像单位(pixel)
		PixelCord_MaskCenter = (self.pixel_array_shape//2 - Distance_MaskCenter_ReconCenter/pixel_spacing,
								self.pixel_array_shape//2 - recon_center_cord[1]           /pixel_spacing)
		PixelDistance_MaskRadius = self.TableIntrinsicCircleRadius / pixel_spacing

		# print(f"Pixel Mask Param: center {PixelCord_MaskCenter}, radius {PixelDistance_MaskRadius}")
		# print(f"Pixel Dist Param: MaskCenter_ReconCenter {Distance_MaskCenter_ReconCenter} ReconCenter_ScanBed {Distance_ReconCenter_ScanBed} MaskCenter_ScanBed {Distance_MaskCenter_ScanBed}")
		# 执行mask
		for x in range(self.pixel_array_shape):
			for y in range(self.pixel_array_shape):
				if (x-PixelCord_MaskCenter[0])**2+(y-PixelCord_MaskCenter[1])**2 > PixelDistance_MaskRadius**2:
					pixel_array[x][y] = self.pad_val
		
		return pixel_array

	def transform(self, results: Dict) -> Dict:
		# 某些没有完整dcm序列的病例不可以进行床体移除，因为其所依赖的相关Metadata不存在。
		if results['dcm_meta']:
			results['img'] = self.process(
				pixel_array=results['img'], 
				table_height=results['dcm_meta']['00181130']['Value'][0], 
				recon_center_cord=results['dcm_meta']['00431031']['Value'], 
				pixel_spacing=results['dcm_meta']['00280030']['Value']
			)
		return results


@TRANSFORMS.register_module()
class RadialStretch(BaseTransform):
	def __init__(self,
				 CornerFactor=1, 
				 GlobalFactor=1, 
				 in_array_shape:tuple=None,
				 direction:str="out", 
				 mmseg_stretch_seg_map:bool=True,
				 stretch_num_workers:int=8):
		assert CornerFactor>=0 and GlobalFactor>=1, "[Dataset] Projection Map Init Error: CornerFactor must >=0, GlobalFactor must >=1"
		assert in_array_shape[0]==in_array_shape[1], "[Dataset] Projection Map Init Error: input image must be square"
		assert direction in ['out', 'in'], "[Dataset] Stretch Direction can only be out or in. Out mean Stretch out to a square. In is its reverse operation"
		
		self.CornerFactor = CornerFactor            # 控制边角拉伸的强度
		self.GlobalFactor = GlobalFactor            # 控制全局放大的强度
		self.in_array_shape = in_array_shape        # 输入矩阵的尺寸
		self.direction = direction                  # 控制拉伸或反拉伸, out为拉伸向外
		self.mmseg_stretch_seg_map = mmseg_stretch_seg_map      # 是否拉伸标签图
		self.stretch_num_workers = stretch_num_workers # 自带的多进程拉伸时的进程数
		self._cache_map()


	def _cache_map(self):
		print_log("[Dataset] 正在缓冲拉伸映射矩阵", "current", logging.INFO)
		# 输出矩阵对应每个点都有一个映射坐标
		map_height, map_width = self.in_array_shape
		self.proj_map = np.zeros(shape=(map_height, map_width, 2), dtype=np.uint16)
		# 遍历每个像素, 生成映射矩阵
		for y in range(map_height):
			for x in range(map_width):
				self.proj_map[y, x] = self.CoordinateMapping(y, x)
		self.proj_map.setflags(write=False) # 锁定映射矩阵
		print_log("[Dataset] 已缓冲拉伸映射矩阵", "current", logging.INFO)

	# 计算该极角的拉伸倍数, 仅适用于正方形输入输出。
	# 输入的是当前映射矩阵的极角, 弧度制。
	# 输出该极角下映射矩阵的极径拉伸倍数，应当在外部与当前映射矩阵的极径相乘，得到source的极径。
	def stretch_factor(self, map_radians):
		# 输入为弧度制
		map_radians = abs(map_radians) % (math.pi/2) # 以90为周期，关于Y轴对称

		# Deprecated: 线性角度映射，有突变点
		# if angle > 45:
		#     angle = 90 - angle  # 周期内中心对称

		# 渐进Cos角度映射，周期内中心对称，但对称模式改变。
		# 设立直角坐标系，X轴及X轴上方有效，X轴代表源角度，Y轴代表映射角度（输出至形变参量的计算）
		# 
		angle = (math.pi/8) * (1 - math.cos(4*map_radians))

		# 形变参量
		radial_factor = 1 / (math.cos(angle)**self.CornerFactor)
		if self.direction == 'out':     # 图像拉伸方向为向外
			radial_factor = 1 / radial_factor
			global_factor = 1 / self.GlobalFactor
		elif self.direction == 'in':    # 图像拉伸方向为向内
			radial_factor = radial_factor
			global_factor = self.GlobalFactor
		# 最终缩放参数 = 该方向上的形变 * 整体缩放
		factor = radial_factor * global_factor
		
		return factor

	# 输入处理后矩阵索引，返回源矩阵索引
	def CoordinateMapping(self, map_Y, map_X):
		# XY均由0开始计数
		# 数组存储图片时，原点位于左上角，这里将Y轴坐标反置，使原点移动至左下角
		true_map_y = self.in_array_shape[0] - map_Y
		# 映射矩阵极点默认为映射矩阵中心
		map_center_y, map_center_x = self.in_array_shape[0]/2, self.in_array_shape[1]/2
		# 输入图像极点默认为输入图像中心
		source_center_y, source_center_x = self.in_array_shape[0]/2, self.in_array_shape[1]/2
		# 获取该点在映射矩阵中的极坐标
		radius, angle = rectangular_to_polar(map_X, true_map_y, map_center_y, map_center_x)
		# 计算该极角的拉伸倍数
		stretch_factor_of_this_angle = self.stretch_factor(angle)
		# 转换回直角坐标, 寻找源坐标
		source_x, true_source_y = polar_to_rectangular(radius*stretch_factor_of_this_angle, angle, source_center_x, source_center_y)
		# 四舍五入, 坐标限制
		source_x, true_source_y = np.clip(round(source_x), 1, self.in_array_shape[1]), np.clip(round(true_source_y), 1, self.in_array_shape[0])
		# Y轴坐标反置的恢复，顺带恢复到索引值域
		source_y = self.in_array_shape[0] - true_source_y
		# X轴恢复到索引值域
		source_x -= 1
		return source_y, source_x

	# 单输入执行拉伸
	def stretch(self, image_matrix):
		# 当参数等效为无拉伸时，直接返回
		if self.CornerFactor==0 and self.GlobalFactor==1:
			return image_matrix
		
		out_shape = self.proj_map.shape[:-1]
		# numpy映射比tensor快一个数量级以上
		# 创建与输入矩阵相同大小的零矩阵
		if isinstance(image_matrix, torch.Tensor):
			if image_matrix.dtype == torch.uint8:
				stretched_matrix = np.zeros(out_shape, dtype=np.uint8)
			else:
				stretched_matrix = np.zeros(out_shape, dtype=np.float32)
		elif isinstance(image_matrix, np.ndarray):
			stretched_matrix = np.zeros(out_shape, dtype=image_matrix.dtype)
		else:
			raise RuntimeError(f"Stretch get unsupported type: {type(stretched_matrix)}")

		# 映射
		map_coordinates(image_matrix, self.proj_map.transpose(2,0,1), output=stretched_matrix, mode='constant', cval=0, prefilter=True)

		if isinstance(image_matrix, torch.Tensor):
			stretched_matrix = torch.from_numpy(stretched_matrix).to(dtype=image_matrix.dtype,
																	 non_blocking=True)
		return stretched_matrix

	# 多进程拉伸
	def multiprocess_stretch(self, image_matrix_list:list | torch.Tensor | np.ndarray) -> list:
		# image_matrix:(B, H, W)
		assert type(image_matrix_list) in [list, torch.Tensor, np.ndarray], f"[Dataset] Projection Typr Error, got {type(image_matrix_list)}"
		assert image_matrix_list[0].shape==self.in_array_shape, f"[Dataset] Projection Error: image shape not match, image{image_matrix_list[0].shape} vs map{self.in_array_shape}"
		if isinstance(image_matrix_list[0], torch.Tensor):
			assert image_matrix_list[0].device == torch.device('cpu'), f"[Dataset] Multiprocess Error: image must be on cpu and then can be passed to subprocess, got {image_matrix_list[0].device}"

		if not hasattr(self, "MULTIPROCESS_POOL"):
			# # 创建共享内存, 该方法最终确定下来起了反效果, /(ㄒoㄒ)/~~
			# self.SharedMemoryBuffer_proj_map = SharedMemory(name=r'shared_proj_map', create=True, size=self.array_shape[0]*self.array_shape[1]*2*2) # 2个uint16, 一个uint16=2字节
			# np_buffer = np.frombuffer(self.SharedMemoryBuffer_proj_map.buf, dtype=np.uint16).reshape(*self.array_shape, 2)
			# np.copyto(np_buffer, self.proj_map)
			# 初始化进程池
			self._init_pool(self.stretch_num_workers)
			self.MULTIPROCESS_POOL: Pool

		if isinstance(image_matrix_list, torch.Tensor):
			image_matrix_list = [image_matrix_list[i].squeeze().cpu().numpy() for i in range(len(image_matrix_list))]
		elif isinstance(image_matrix_list, np.ndarray):
			image_matrix_list = [image_matrix_list[i].squeeze() for i in range(len(image_matrix_list))]
		else:
			image_matrix_list = [image_matrix_list[i] for i in range(len(image_matrix_list))]

		stretched_matrix_iterator = self.MULTIPROCESS_POOL.map(self.stretch, image_matrix_list)
		return stretched_matrix_iterator

	# 进程池初始化, 对象共享进程池
	@classmethod
	def _init_pool(cls, num_workers:int=cpu_count(), pool_initializer=None):
		cls.MULTIPROCESS_POOL = Pool(num_workers, pool_initializer)

	# 反归一化
	@staticmethod
	def reverse_norm(image_matrix, mean, std):
		return image_matrix * std + mean

	# MMSegmentation 接口
	def transform(self, results: dict) -> dict:
		results['img'] = self.stretch(results['img'])
		if 'gt_seg_map' in results and self.mmseg_stretch_seg_map:
			results['gt_seg_map'] = self.stretch(results['gt_seg_map'])
		return results

	# 多进程时, 手动释放共享内存, 否则多次执行后页面文件可能溢出, 原因未明
	def __del__(self):
		if hasattr(self, "multiprocess_pool"):
			self.multiprocess_pool.close()
			self.multiprocess_pool.join()
		import gc
		gc.collect()


# 使用mmengine的ManagerMixin创建全局类
class RadialStretch_SharedWrapper(ManagerMixin):
	def __init__(self, name, RadialStretchConfig):
		super().__init__(name)
		self.RadialStretch = RadialStretch(**RadialStretchConfig)

	def __getattr__(self, name):
		if name in ['stretch', 'multiprocess_stretch']:
			return getattr(self.RadialStretch, name)
		else:
			return getattr(self, name)


@TRANSFORMS.register_module()
class Distortion(BaseTransform):
	def __init__(self, 
				 global_rotate:float,
				 amplitude:float, 
				 frequency:float, 
				 grid_dense:int, 
				 in_array_shape:tuple,
				 refresh_interval:int,
				 const:bool=False) -> None:
		self.global_rotate = global_rotate  # 随机全局旋转
		# 使用正弦函数建立极径与扭曲角之间的关系
		self.amplitude = amplitude  # 振幅
		self.frequency = frequency  # 频率
		self.img_shape = in_array_shape  # 输入矩阵的尺寸
		self.grid_dense = grid_dense  # 网格密度
		self.refresh_interval = refresh_interval  # 映射矩阵刷新间隔
		self.refresh_counter = 0
		self.const = const			# 控制是否要固定AF参数
		self.tform = PiecewiseAffineTransform()

		self.refresh_affine_map()       # 初始化映射矩阵
		super().__init__()

	# 为分段仿射变换生成映射矩阵
	# 有时候为了固定AF参数，需要使用const来关闭随机参数
	def refresh_affine_map(self):
		src_cols = np.linspace(0, self.img_shape[0], self.grid_dense)
		src_rows = np.linspace(0, self.img_shape[1], self.grid_dense)
		# 构建网格, src_rows, src_cols形状为(grid_dense, grid_dense)
		# src_rows为网格在源图中的行坐标, src_cols为网格在源图中的列坐标
		src_rows, src_cols = np.meshgrid(src_rows, src_cols)
		# src包括所有网格点在源图像中的坐标，形状为(grid_dense, grid_dense, 2)
		src = np.stack([src_cols, src_rows], axis=2)
		dst = np.zeros_like(src)

		amplitude = (1 if self.const else random.random()*2-1) * self.amplitude
		frequency = (1 if self.const else random.random()*2-1) * self.frequency
		global_rotate = (1 if self.const else random.random()*2-1) * self.global_rotate
		
		for x in range(self.grid_dense):
			for y in range(self.grid_dense):
				# index转标准坐标
				y = self.grid_dense - y - 1
				# 直角坐标转极坐标
				radius, angle = rectangular_to_polar(x, y, self.grid_dense//2, self.grid_dense//2)
				# 映射网格坐标系转换为源坐标系
				radius *= self.img_shape[0] / self.grid_dense
				# 极径扭曲
				angle += math.pi/8 * amplitude * np.sin(radius/self.img_shape[0] * frequency * 2 * math.pi)
				# 全局旋转
				angle += global_rotate * math.pi/2
				# 极坐标转换为直角坐标
				src_x, src_y = polar_to_rectangular(radius, angle, self.img_shape[0]/2, self.img_shape[1]/2)
				# 标准坐标转index
				src_y = self.img_shape[0] - src_y - 1
				y = self.grid_dense - y - 1
				# 存入dst
				dst[x, y, :] = (src_x, src_y)

		self.tform.estimate(src.reshape(-1,2), dst.reshape(-1,2))

	@classmethod
	def _init_pool(cls):
		cls.p = Pool(int(cpu_count()*0.8))


	def multiprocess_distort(self, Imgarray_list:list) -> List[np.ndarray]:
		if not hasattr(self, 'p'):
			self._init_pool()
		distorted_imgs = []
		fetcher = self.p.imap(self.distort, Imgarray_list)
		for distorted_img in tqdm(fetcher, desc='MultiProcess Distortion', total=len(Imgarray_list)):
			distorted_imgs.append(distorted_img)

		if not self.const:
			self.refresh_counter += 1
			if self.refresh_counter % self.refresh_interval == 0:
				self.refresh_affine_map()
		
		return distorted_imgs

	# 执行
	def distort(self, Imgarray:np.ndarray):
		return warp(image=Imgarray, 
					inverse_map=self.tform, 
					order=3,
					preserve_range=True)

	# MMSegmentation 接口
	def transform(self, results: dict) -> dict:
		# 在开始时或每隔一段时间，刷新映射矩阵
		if not self.const:
			if self.refresh_counter % self.refresh_interval == 0:
				self.refresh_affine_map()
		
		results['img'] = warp(image=results['img'], 
							  inverse_map=self.tform, 
							  order=1,
							  preserve_range=True)
		if 'gt_seg_map' in results:
			results['gt_seg_map'] = warp(image=results['gt_seg_map'], 
								inverse_map=self.tform, 
								order=0,
								preserve_range=True)
		
		self.refresh_counter += 1
		return results

# Original Images min: -2,000, max: 4,095, mean: 10.87, std: 1,138.37
# after clip: min: 0, max: 4,095, mean: 561.54, std: 486.59
@TRANSFORMS.register_module()
class RangeClipNorm(BaseTransform):
	def __init__(self, input_shape, contrast, brightness, norm_method:str, mean=None, std=None) -> None:
		assert norm_method in ['const', 'inst', 'hist'], "[Dataset] CTImageEnhance Augmentation: norm_method must be 'const' or 'inst'"
		if mean is None or std is None: 
			assert norm_method!='const', "[Dataset] CTImageEnhance Augmentation: mean and std must be provided when norm_method is 'const'"
			self.mean = self.std = None
		else:
			self.mean = (mean+brightness)*contrast
			self.std = std*contrast
		self.norm_method = norm_method

		if norm_method=='hist':
			self.nbins = 256
			self.mask = self.create_circle_in_square(input_shape[0], input_shape[0]//3)

		self.contrast = contrast
		self.brightness = brightness

		if (brightness!=0 or contrast!=1) and norm_method=='const':
			print(f'\033[93m[Dataset] WARNING: Norm param may be incorrect due to effective range clip\033[0m')
		super().__init__()

	@staticmethod
	def create_circle_in_square(size, radius):
		# 创建一个全0的正方形ndarray
		square = np.zeros((size, size))
		# 计算中心点的坐标
		center = size // 2
		# 计算每个元素到中心的距离
		y, x = np.ogrid[:size, :size]
		mask = (x - center)**2 + (y - center)**2 <= radius**2
		# 如果距离小于或等于半径，将该元素设置为1
		square[mask] = 1
		return square


	def hist_equal(self, Imgarray:np.ndarray):
		assert Imgarray.shape == self.input_shape, f"HistogramEqualization Augmentation: input_shape expected{self.input_shape}, got {Imgarray.shape}"
		return equalize_hist(Imgarray, nbins=self.nbins, mask=self.mask)

	# 定制算法
	def _exec(self, Imgarray:np.ndarray):
		# Range Enhance
		value_range_center = np.clip(2048+self.brightness, 0, 4095)
		value_range_width_half = np.clip(2048/self.contrast, 0, 2048)
		Imgarray = np.clip(Imgarray, 
						value_range_center-value_range_width_half, 
						value_range_center+value_range_width_half).astype(np.float32)
		# Normalize
		if self.norm_method == 'const':
			Imgarray = (Imgarray - self.mean) / self.std
		elif self.norm_method == 'inst':
			Imgarray = (Imgarray - Imgarray.mean()) / Imgarray.std()
		elif self.norm_method == 'hist':
			Imgarray = equalize_hist(Imgarray, nbins=self.nbins, mask=self.mask)

		return Imgarray
	
	# MMSegmentation 接口
	def transform(self, results: dict) -> dict:
		ImgNdarray = results['img']
		assert isinstance(ImgNdarray, np.ndarray), "CTImageEnhance Augmentation: input img must be a numpy ndarray"
		ImgNdarray = self._exec(ImgNdarray)
		# print('enhenced:', ImgNdarray.shape, ImgNdarray.min(), ImgNdarray.max(), ImgNdarray.mean(), ImgNdarray.std(), ImgNdarray.dtype)
		results['img'] = ImgNdarray
		results['img_shape'] = ImgNdarray.shape
		return results


@TRANSFORMS.register_module()
class GaussianBlur(BaseTransform):
	def __init__(self, sigma, radius) -> None:
		self.sigma = sigma
		self.radius = radius
		super().__init__()
	
	def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:
		results['img'] = gaussian_filter(results['img'], 
										 sigma=self.sigma, 
										 order=3,
										 radius=self.radius,
										 )
		return results



@TRANSFORMS.register_module()
class OriShapeOverride(BaseTransform):
	def __init__(self, ori_shape:tuple) -> None:
		super().__init__()
		self.ori_shape = ori_shape

	def transform(self, results: dict) -> dict:
		results['img_shape'] = self.ori_shape
		results['ori_shape'] = self.ori_shape
		results['scale_factor'] = (1,1)
		return results



@HOOKS.register_module()
class CTSegVisualizationHook(SegVisualizationHook):
    def __init__ (self, reverse_stretch=None, lmdb_backend_proxy:LMDB_MP_Proxy=None, **kwargs):
        super().__init__(**kwargs)

        if lmdb_backend_proxy:
            if isinstance(lmdb_backend_proxy, ConfigDict):
                lmdb_backend_proxy = TRANSFORMS.build(lmdb_backend_proxy)
            self.lmdb_service:LMDB_DataBackend = lmdb_backend_proxy()


        if reverse_stretch:
            if RadialStretch_SharedWrapper.check_instance_created('shared_stretch'):
                self.reverse_stretch=RadialStretch_SharedWrapper.get_instance('shared_stretch')
            else:
                self.reverse_stretch = RadialStretch_SharedWrapper.get_instance('shared_stretch', RadialStretchConfig=reverse_stretch)

    def source_img(self, output:SegDataSample):
        if hasattr(self, 'lmdb_service'):
            (path, meta_buffer, img_buffer) = self.lmdb_service.fetch_data(output.img_path)
            (meta, pixel) = LMDB_DataBackend.decompress(meta_buffer, img_buffer)
        else:
            pixel = np.load(output.img_path)
        
        return (np.clip(pixel.reshape(512,512,1),0,4095)//16).astype(np.uint8)

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """

        if self.draw is False or mode == 'train':
            return

        # mmseg visualization
        if self.every_n_inner_iters(batch_idx, self.interval):
            # 重载原始图像
            img_list = []
            for output in outputs:
                img_list.append(self.source_img(output))
            
            if hasattr(self, 'reverse_stretch'):
                # 推理预测 反拉伸
                original_pred_sem_seg_shape = outputs[0].pred_sem_seg.data.shape
                reverse_stretch_queue = [output.pred_sem_seg.data.squeeze().cpu().to(torch.uint8) 
                                         for output in outputs]
                
                # 执行多进程反拉伸并解包
                sample_iterator = self.reverse_stretch.multiprocess_stretch(reverse_stretch_queue)
                assert len(sample_iterator) == len(outputs)
                for batch_index in range(len(outputs)):
                    outputs[batch_index].pred_sem_seg.data = \
                        sample_iterator[batch_index].reshape(*original_pred_sem_seg_shape)

            for i, output in enumerate(outputs):
                img_path = output.img_path
                img = img_list[i]
                
                window_name = f'{mode}_{osp.basename(img_path)}'
                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)




if __name__ == '__main__':
	dcm = pydicom.dcmread(r".\img\柏家荣\ImageFileName000.dcm")

	from mmseg.CQK_Med import LMDB_MP_Proxy
	lmdb_args = {'dataset_root': "../2023_Med_CQK", 
				'lmdb_path': "../2023_Med_CQK/lmdb_database", 
				'mode': "normal"}
	backend = LMDB_MP_Proxy.get_instance('test', lmdb_args=lmdb_args)
	backend = LMDB_DataBackend(**lmdb_args)
	backend = LMDB_MP_Proxy('test', lmdb_args)
	backend.database_test()






