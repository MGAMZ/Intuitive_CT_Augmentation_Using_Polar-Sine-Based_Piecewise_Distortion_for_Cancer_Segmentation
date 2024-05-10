import os, tqdm, time, pdb, pickle, warnings
from typing import Tuple, Dict, Union, List
from multiprocessing import Pool, cpu_count

import pydicom, nrrd
import torch
import numpy as np


'''
数据集抽象数据格式：
dict(
	'METADATA': dict(
		'num_Img': int,		# 数据集中所有病例的所有图像数量
		'num_patient': int,	# 数据集中所有病例数量
		'mode': dict(		# 数据集中所有病例的模式分布
			'normal': int,
			'img_only': int,
			'label_only': int,
			'CTEnhanceScanPhaseLost': int,
			'exception': int,
		),
	),
	'%PatientName1': dict(
		'ImgLabelPairStatus': str,  # normal, img_only, label_only, CTEnhanceScanPhaseLost, exception
		'APVV': bool,	# 是否为APVV多方向扫描
		'N': dict(
			'label_Y': float,				# 标注图的人体纵向坐标，读取自nrrd标注文件元数据
			'label_index': int,				# 标注图在原始扫描中的序列号
			'series_Y_range':[float, float]	# 该扫描期的人体纵向坐标范围
			'LabeledImgFileName': str,		# 标注所对应的扫描图像文件名'
			'NumImgFile': int,				# 对应的原始扫描图像个数
			'NumIndexDigitOfImgFile':int	# 图像文件名的序列号位数
			'index_Y_dict': Dict[index: Y]	# 图像文件名的序列号与纵向坐标的映射
		)
		'A' :...,
		'PV':...,
		'V' :dict(),
	),
	'%PatientName2': dict(...),
	'%PatientName3': dict(...),
	...
)
'''

'''
原始文件夹结构：
root
|	|---img
|	|	|---name1
|	|	|	|---img1.dcm
|	|	|	|---img2.dcm
|	|	|	|---...
|	|	|	|---img1.png
|	|	|	|---img2.png
|	|	|	|---...
|	|	|	|---A(optional)
|	|	|	|	|---...
|	|	|	|---PV(optional)
|	|	|	|	|---...
|	|	|	|---V(optional)
|	|	|	|	|---...
|	|	|---name2
|	|	|---...
|	|---label
|	|	|---name1
|	|	|	|---image.nrrd
|	|	|	|---mask.nrrd
|	|	|	|---A(optional)
|	|	|	|	|---...
|	|	|	|---PV(optional)
|	|	|	|	|---...
|	|	|	|---V(optional)
|	|	|	|	|---...
|	|	|---name2
|	|	|---...

'''


# 从NRRD标注文件中读取标注图的人体纵向坐标
def Label_Y_from_NRRD(nrrd_path:str):
	data, header = nrrd.read(nrrd_path)
	return header['space origin'][2]


def FindImg_according_to_Label_Y(ImgFolderPath:str, Label_Y:float):
	dcmList = [file for file in os.listdir(ImgFolderPath) if file.endswith('.dcm')]
	for dcm_file in dcmList:
		dcm = pydicom.dcmread(os.path.join(ImgFolderPath, dcm_file))
		dcm_Y = dcm.get('SliceLocation')	# (0020, 1041) Slice Location                      DS: '-184.656'
		# 绝对值差不超过0.3的就认为是同一张切片
		if abs(dcm_Y - Label_Y) < 0.5:
			filename = os.path.basename(dcm_file)
			LabeledImgIndex = int(filename.split('.')[0][-3:] if len(dcmList)<=1000 else filename.split('.')[0][-4:])
			return filename, LabeledImgIndex
	else:
		print(f'TargetSlice_Y: {Label_Y} not found in {ImgFolderPath}')


class CQKGastricCancerCT:
	def __init__(self, root:str, metadata_ckpt=None, *args, **kwgs) -> None:
		self.root = root
		self.split_ratio = {'train': 0.75, 'val':0.1, 'test':0.15}	# train/val/test 比例\
		assert sum(self.split_ratio.values()) == 1.0, 'split_ratio must sum to 1.0'
		if metadata_ckpt is None:
			self.dataset = dict()
			self._AnalyzeFolderStructure()	# 分析数据集文件组织结构
			save_ckpt_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
			# 把self.dataset整个dict对象序列化存储到本地
			with open(f'{save_ckpt_name}.pickle', 'wb') as f:
				pickle.dump(self.dataset, f)
		elif isinstance(metadata_ckpt, str):
			with open(metadata_ckpt, 'rb') as f:
				self.dataset = pickle.load(f)
		elif isinstance(metadata_ckpt, dict):
			self.dataset = metadata_ckpt
		else:
			raise TypeError(f'metadata_ckpt must be str or dict, but got {type(metadata_ckpt)}')

		self.EmptyLabel = np.zeros((512,512), dtype=np.float32)

	def _AnalyzeFolderStructure(self):
		print("未设定索引文件，正在生成索引。")

		# 检查目录下是否有img和label两个文件夹
		if not os.path.exists(os.path.join(self.root, 'img')):
			raise Exception('img folder not found')
		if not os.path.exists(os.path.join(self.root, 'label')):
			raise Exception('label folder not found')
		
		self._PairImgLabelFolder()			# 检查Img和Label文件夹的匹配情况
		print("img和label病例匹配检查完成")
		self._IdentifyScanDirection()		# 识别各病例的扫描方向
		print("各病例方向匹配完成")
		self._register_ScanImgLabel_info()	# 初始化每次CT扫描和标注的对应信息、扫描数量、标注坐标等信息
		print("病例注册完成")
		self._generate_METADATA()			# 记录数据集整体信息
		print("数据集整体参数统计完成")

	def _PairImgLabelFolder(self):
		# 检查img和label文件夹下子文件夹的匹配情况
		img_folders = os.listdir(os.path.join(self.root, 'img'))
		label_folders = os.listdir(os.path.join(self.root, 'label'))
		
		# 针对两种情况输出：1. img有的子文件夹label没有 2. label有的子文件夹img没有
		self.ImgOnlyName, self.LabelOnlyName = [], []
		self.NormalImgLabelPairsName = []
		for img_folder in img_folders:
			if img_folder not in label_folders:
				self.ImgOnlyName.append(img_folder)
				self.dataset[img_folder] = {'ImgLabelPairStatus': 'img_only'}
			else:
				self.NormalImgLabelPairsName.append(img_folder)
				self.dataset[img_folder] = {'ImgLabelPairStatus': 'normal'}
		for label_folder in label_folders:
			if label_folder not in img_folders:
				self.LabelOnlyName.append(label_folder)
				self.dataset[label_folder] = {'ImgLabelPairStatus': 'label_only'}
		print('\033[33m', '\n', f'img folders not in label folders({len(self.ImgOnlyName)}):\n {self.ImgOnlyName}', '\033[0m')
		print('\033[33m', '\n', f'label folders not in img folders({len(self.LabelOnlyName)}):\n {self.LabelOnlyName}', '\033[0m')
		print(f'\nNormalImgLabelPairs NUM: {len(self.NormalImgLabelPairsName)}')

	def _IdentifyScanDirection(self):
		# 如果子文件夹下存在A、PV、V子文件夹，则需要进一步分类。
		# 遍历三种情况的Name文件夹，查询旗下是否有子目录
		self.APVVName = []
		for name in self.NormalImgLabelPairsName:
			PatientImgSubDir = [f.name for f in os.scandir(os.path.join(self.root, 'img', name)) if f.is_dir()]
			PatientLabelSubDir = [f.name for f in os.scandir(os.path.join(self.root, 'label', name)) if f.is_dir()]
			if len(PatientImgSubDir) != 0 or len(PatientLabelSubDir) != 0:
				self.APVVName.append(name)
				if not 'A' in PatientImgSubDir or not 'PV' in PatientImgSubDir:	# 若为APVV，则代表增强CT扫描，label文件夹会有A\PV标注，但原始图像文件夹不一定有相同的文件夹形式
					self.dataset[name]['ImgLabelPairStatus'] = 'CTEnhanceScanPhaseLost'	# 代表增强CT的至少有一个扫描期的图像文件夹丢失
				else:
					for direction in PatientImgSubDir + PatientLabelSubDir:
						self.dataset[name][direction] = dict()
		for name in self.ImgOnlyName:
			PatientSubDir = [f.name for f in os.scandir(os.path.join(self.root, 'img', name)) if f.is_dir()]
			if len(PatientSubDir) != 0:
				self.APVVName.append(name)
				for direction in PatientSubDir:
					self.dataset[name][direction] = dict()
		for name in self.LabelOnlyName:
			PatientSubDir = [f.name for f in os.scandir(os.path.join(self.root, 'label', name)) if f.is_dir()]
			if len(PatientSubDir) != 0:
				self.APVVName.append(name)
				for direction in PatientSubDir:
					self.dataset[name][direction] = dict()
		
		for name in self.APVVName:
			self.dataset[name]['APVV'] = True
		print(f'\nAPVV NUM: {len(self.APVVName)}')

		# 如果结束时没有‘direction’键的，则为默认方向，为其添加‘N’键
		for name in self.dataset:
			if name not in self.APVVName:
				self.dataset[name]['APVV'] = False
				self.dataset[name]['N'] = dict()

	@classmethod
	def _multiprocess_init(cls):
		cls.MULTIPROCESS_POOL = Pool(cpu_count())

	def register_one_patient(self, patient):
		patient_dict = self.dataset[patient]
		if patient_dict['ImgLabelPairStatus'] == 'exception':
			return
		for phase in ['N','A','PV','V','D']:	# 'V'平衡期未作标注
			if phase not in patient_dict: continue
			ImgFolderPath = os.path.join(self.root, 'img', patient, phase) if phase!='N' else os.path.join(self.root, 'img', patient)
			LabelFolderPath = os.path.join(self.root, 'label', patient, phase) if phase!='N' else os.path.join(self.root, 'label', patient)

			# label定位到dcm序列中的位置
			# 若为label_only或是只存在于img文件夹中的CT增强扫描期，则无法定位其在原始扫描中的序列号
			try:
				label_Y = Label_Y_from_NRRD(os.path.join(LabelFolderPath, 'mask.nrrd'))
				LabeledImgFileName, LabeledImgIndex = FindImg_according_to_Label_Y(ImgFolderPath, label_Y)
				index_Y_dict, min_Y, max_Y = self._Y_index_from_DCM_series(ImgFolderPath)
			except Exception as e:
				if not isinstance(e, FileNotFoundError):
					print(str(e))
				label_Y = None
				LabeledImgFileName, LabeledImgIndex = None, None
				index_Y_dict, min_Y, max_Y = None, None, None

			# dcm序列元数据获取
			# 只有存在完整dcm序列时，才能获取可靠元数据
			try:
				dcm_meta = os.path.join(ImgFolderPath, os.listdir(ImgFolderPath)[0])
				dcm_meta = pydicom.dcmread(dcm_meta)
				patient_dict[phase]['metadata'] = {
					'table_height': 	dcm_meta.get(('0018', '1130')).value,
					'recon_center_cord':dcm_meta.get(('0043', '1031')).value,
					'pixel_spacing':	dcm_meta.get(('0028', '0030')).value,
				}
			except Exception as e:
				if not isinstance(e, FileNotFoundError):
					print("Metadata fetch failed for", e)
				patient_dict[phase]['metadata'] = {}

			patient_dict[phase]['label_Y'] = label_Y
			patient_dict[phase]['LabeledImgFileName'] = LabeledImgFileName
			patient_dict[phase]['label_index'] = LabeledImgIndex
			patient_dict[phase]['series_Y_range'] = [min_Y, max_Y]
			patient_dict[phase]['index_Y_dict'] = index_Y_dict

			# 获取原始扫描图像数量
			if patient_dict['ImgLabelPairStatus'] != 'label_only':
				try:
					patient_dict[phase]['NumImgFile'] = len([f for f in os.listdir(ImgFolderPath) if f.endswith('.npy')])
				except:
					patient_dict[phase]['NumImgFile'] = 0	# 若对应img文件夹不存在，则赋0
			# 若为LabelOnly模式，则其NumImgFile为1，由数据集中label文件夹内的image.nrrd提供，且其方向与mask.nrrd是相同的
			else:
				patient_dict[phase]['NumImgFile'] = 0
			
			# 对于有原始扫描图像的扫描期，获取原始扫描图像文件名的序列号位数
			if patient_dict[phase]['NumImgFile'] != 0:
				# 取出一个文件判断这一扫描期的结尾编号有几位数字
				example_file_name = os.listdir(ImgFolderPath)[0]
				patient_dict[phase]['NumIndexDigitOfImgFile'] = len(example_file_name.split('.')[0])-13	# 去除前缀'ImageFileName'
		
		return patient, patient_dict

	def _register_ScanImgLabel_info(self):
		if not hasattr(self, 'MULTIPROCESS_POOL'):
			self._multiprocess_init()
		result_fetcher = self.MULTIPROCESS_POOL.imap_unordered(self.register_one_patient, self.dataset.keys())
		for patient_name, patient_dict in tqdm.tqdm(result_fetcher, desc='Multiprocess Registering', total=len(self.dataset)):
			self.dataset[patient_name] = patient_dict
		self.MULTIPROCESS_POOL.close()
		self.MULTIPROCESS_POOL.join()

		# 	if self.dataset[patient]['ImgLabelPairStatus'] == 'exception':
		# 		continue
		# 	for phase in ['N','A','PV','V','D']:	# 'V'平衡期未作标注
		# 		if phase not in self.dataset[patient]: continue
		# 		ImgFolderPath = os.path.join(self.root, 'img', patient, phase) if phase!='N' else os.path.join(self.root, 'img', patient)
		# 		LabelFolderPath = os.path.join(self.root, 'label', patient, phase) if phase!='N' else os.path.join(self.root, 'label', patient)

		# 		# label定位到dcm序列中的位置
		# 		# 若为label_only或是只存在于img文件夹中的CT增强扫描期，则无法定位其在原始扫描中的序列号
		# 		try:
		# 			label_Y = Label_Y_from_NRRD(os.path.join(LabelFolderPath, 'mask.nrrd'))
		# 			LabeledImgFileName, LabeledImgIndex = FindImg_according_to_Label_Y(ImgFolderPath, label_Y)
		# 			index_Y_dict, min_Y, max_Y = self._Y_index_from_DCM_series(ImgFolderPath)
		# 		except Exception as e:
		# 			if not isinstance(e, FileNotFoundError):
		# 				tqdm.tqdm.write(str(e))
		# 			label_Y = None
		# 			LabeledImgFileName, LabeledImgIndex = None, None
		# 			index_Y_dict, min_Y, max_Y = None, None, None

		# 		# dcm序列元数据获取
		# 		# 只有存在完整dcm序列时，才能获取可靠元数据
		# 		try:
		# 			dcm_meta = ImgFolderPath[0]
		# 			dcm_meta = pydicom.dcmread(dcm_meta)
		# 			self.dataset[patient][phase]['metadata'] = {
		# 				'table_height': 	dcm_meta.get(('0018', '1130')),
		# 				'recon_center_cord':dcm_meta.get(('0043', '1031')),
		# 				'pixel_spacing':	dcm_meta.get(('0028', '0030')),
		# 			}
		# 		except Exception as e:
		# 			print("Metadata fetch failed for", e)
		# 			self.dataset[patient][phase]['metadata'] = {}

		# 		self.dataset[patient][phase]['label_Y'] = label_Y
		# 		self.dataset[patient][phase]['LabeledImgFileName'] = LabeledImgFileName
		# 		self.dataset[patient][phase]['label_index'] = LabeledImgIndex
		# 		self.dataset[patient][phase]['series_Y_range'] = [min_Y, max_Y]
		# 		self.dataset[patient][phase]['index_Y_dict'] = index_Y_dict

		# 		# 获取原始扫描图像数量
		# 		if self.dataset[patient]['ImgLabelPairStatus'] != 'label_only':
		# 			try:
		# 				self.dataset[patient][phase]['NumImgFile'] = len([f for f in os.listdir(ImgFolderPath) if f.endswith('.npy')])
		# 			except:
		# 				self.dataset[patient][phase]['NumImgFile'] = 0	# 若对应img文件夹不存在，则赋0
		# 		# 若为LabelOnly模式，则其NumImgFile为1，由数据集中label文件夹内的image.nrrd提供，且其方向与mask.nrrd是相同的
		# 		else:
		# 			self.dataset[patient][phase]['NumImgFile'] = 0
				
		# 		# 对于有原始扫描图像的扫描期，获取原始扫描图像文件名的序列号位数
		# 		if self.dataset[patient][phase]['NumImgFile'] != 0:
		# 			# 取出一个文件判断这一扫描期的结尾编号有几位数字
		# 			example_file_name = os.listdir(ImgFolderPath)[0]
		# 			self.dataset[patient][phase]['NumIndexDigitOfImgFile'] = len(example_file_name.split('.')[0])-13	# 去除前缀'ImageFileName'

	def _generate_METADATA(self):
		total_NumImgFile = 0
		total_NumLabelFile = 0
		total_mode = {'normal':0, 'img_only':0, 'label_only':0, 'CTEnhanceScanPhaseLost':0, 'exception':0}
		total_patient = len(self.dataset)
		for patient in self.dataset:
			total_mode[self.dataset[patient]['ImgLabelPairStatus']] += 1
			for phase in ['N','A','PV','V']:
				item = self.dataset[patient].get(phase, None)
				if item is not None:
					total_NumImgFile += item['NumImgFile']
					if item.get('LabeledImgFileName', None) is not None:
						total_NumLabelFile += 1
		self.dataset['METADATA'] = {
			'num_Img': total_NumImgFile,
			'num_patient': total_patient,
			'mode': total_mode,
			'num_label': total_NumLabelFile,
		}

	def _UpdateDatasetLength(self):
		if self.ensambled_img_group:
			raise NotImplementedError('ensambled_img_group function not implemented')
		
		num_samples = 0
		# 遍历所有病例，统计每个病例的样本数量
		for patient in self.dataset:
			if patient == 'METADATA': continue
			num_samples += self.GetPatientNumSamples(patient)['NumSamples']
		
		self.dataset_length = num_samples

	def _GetPath(self, patient_name:str, type:str, phase:str, index:str=None):
		'''
			patient_name: 病例姓名
			type: 取出图像、标注、或是标注附带的图像	对应：'img', 'label', 'img_in_label'
			phase: N/A/PV/V/D CT增强扫描期
			index: 图像扫描延人体纵轴的序列号，从0开始
		
		'''
		assert type in ['img', 'label', 'img_in_label']
		assert phase in ['N', 'A', 'PV', 'V', 'D']
		if index is not None: assert type == 'img'

		if type == 'img':
			if self.dataset[patient_name][phase]['NumIndexDigitOfImgFile'] == 4:
				filename = f'ImageFileName{index:04d}.npy'
			elif self.dataset[patient_name][phase]['NumIndexDigitOfImgFile'] == 3:
				filename = f'ImageFileName{index:03d}.npy'
			else:
				pdb.set_trace()
				raise RuntimeError('严重错误：逻辑Exception！原始扫描图像序列号应由3位或4位数字组成')
		elif type == 'label':
			filename = 'mask.npy'
		elif type == 'img_in_label':	# 在某些时候，标注文件不存在原图序列，则由label中image文件提供唯一原图
			type = 'label'
			filename = 'image.npy'
		else:
			raise RuntimeError('严重错误：逻辑Exception！')

		# 生成图像路径
		if phase == 'N':
			return os.path.join(self.root, type, patient_name, filename)
		else:
			return os.path.join(self.root, type, patient_name, phase, filename)

	@staticmethod
	def _Y_index_from_DCM_series(dcm_path_root:str)-> Tuple[Dict, int, int]:
		assert os.path.isdir(dcm_path_root), 'dcm_path_root must be a series of dcm files'
		
		dcmList = [file for file in os.listdir(dcm_path_root) if file.endswith('.dcm')]
		dcmList.sort()
		max_Y = -np.inf
		min_Y = np.inf
		index_Y_dict = {}

		for dcm_file in dcmList:
			dcm = pydicom.dcmread(os.path.join(dcm_path_root, dcm_file))
			dcm_Y = float(dcm.get('SliceLocation'))	# (0020, 1041) Slice Location                      DS: '-184.656'
			filename = os.path.basename(dcm_file)
			ImgIndex = int(filename.split('.')[0][-3:] if len(dcmList)<=1000 else filename.split('.')[0][-4:])
			index_Y_dict[ImgIndex] = dcm_Y
			if dcm_Y > max_Y:
				max_Y = dcm_Y
			if dcm_Y < min_Y:
				min_Y = dcm_Y

		return index_Y_dict, min_Y, max_Y

	def _InitGetItemIndex(self):
		temp_index_list = []	# [img_path, label_path, [patient, phase, index]]

		for patient in self.activated_patient_name:
			if patient == 'METADATA': continue	# 排除dataset中的METADATA项，只对病例进行提取
			for phase in ['N', 'A', 'PV', 'V', 'D']:
				if phase not in self.dataset[patient]: continue	# 病例没有该扫描期
				
				# 使用无监督预训练时，只需要取出所有扫描期的原始图像并生成索引即可
				if self.pretraining:
					if self.dataset[patient]['ImgLabelPairStatus'] == 'label_only':	# 只有标注附带的原始图像，则只取出这一张作为无监督样本即可
							img_path = self._GetPath(patient, 'img_in_label', phase, None)
							temp_index_list.append([img_path, None, [patient, phase, 0]])	# 无监督预训练无需label
					else:
						for ImgIndex in range(self.dataset[patient][phase]['NumImgFile']):
							img_path = self._GetPath(patient, 'img', phase, ImgIndex)
							temp_index_list.append([img_path, None, [patient, phase, ImgIndex]])	# 无监督预训练无需label
					
				# 使用其他训练模式时，需要考虑原始图像和标注文件对于训练流的影响
				else:
					# 可进行基于距离的数据增强的病例
					if self.dataset[patient]['ImgLabelPairStatus'] == 'normal' or self.dataset[patient]['ImgLabelPairStatus'] == 'CTEnhanceScanPhaseLost':
						# 判断该扫描期是否有对应的标注
						if self.dataset[patient][phase]['LabeledImgFileName'] is None: continue
						# 生成正负样本的index索引范围
						PositiveImageIndexStart = self.dataset[patient][phase]['label_index'] - (self.num_positive_img-1)//2
						PositiveImageIndexList = list(range(PositiveImageIndexStart, PositiveImageIndexStart+self.num_positive_img))
						NegativeImageIndexRangeLeft = range(0, self.dataset[patient][phase]['label_index'] - self.minimum_negative_distance)
						NegativeImageIndexRangeRight = range(self.dataset[patient][phase]['label_index'] + self.minimum_negative_distance + 1, self.dataset[patient][phase]['NumImgFile'])
						NegativeImageIndexList = list(NegativeImageIndexRangeLeft)+list(NegativeImageIndexRangeRight)
						# 从NegativeImageIndexList中随机抽取num_negative_img个样本作为负样本
						np.random.shuffle(NegativeImageIndexList)	# 原地操作
						NegativeImageIndexList = NegativeImageIndexList[:self.num_negative_img]

						for ImgIndex in PositiveImageIndexList:
							img_path = self._GetPath(patient, 'img', phase, ImgIndex)
							label_path = self._GetPath(patient, 'label', phase)
							temp_index_list.append([img_path, label_path, [patient, phase, ImgIndex]])
						for ImgIndex in NegativeImageIndexList:
							img_path = self._GetPath(patient, 'img', phase, ImgIndex)
							temp_index_list.append([img_path, None, [patient, phase, ImgIndex]])
					# 由于没有原始扫描图片，无法进行基于距离的数据增强的病例
					elif self.dataset[patient]['ImgLabelPairStatus'] == 'label_only':
						img_path = self._GetPath(patient, 'img_in_label', phase)
						label_path = self._GetPath(patient, 'label', phase)
						temp_index_list.append([img_path, label_path, [patient, phase, 0]])
					elif self.dataset[patient]['ImgLabelPairStatus'] == 'img_only':
						continue	# 只有扫描图像没有标注的病例只能用于无监督预训练

		self.IndexList = temp_index_list	# self.IndexList: [img_path, label_path, [patient, phase, index]]

	def GetPatientNumSamples(self, patient_name:str):
		PatientMETADATA = {
			'NumImgFile': -1,
			'NumLabelFile': -1,
			'ImgLabelPairStatus': None,
			'APVV': None,
		}
		# 尝试读取病例索引
		PatientData = self.dataset.get(patient_name, None)
		if PatientData is None:
			raise Exception(f'Patient {patient_name} not found in dataset index dict')
		
		# 统计病例总图像数、总标注数
		TotalNumImg = 0
		TotalNumLabel = 0
		for phase in PatientData:
			TotalNumImg += PatientData[phase].get('NumImgFile', 0)
			if PatientData[phase].get('LabeledImgFileName', None) is not None:
				TotalNumLabel += 1
		
		# 预训练通常只关心独立的图像数量，采用对比学习时应当再行调整
		if self.pretraining: 
			return {'NumSamples': TotalNumImg ,'NumLabel': TotalNumLabel}
		
		# 进行分割训练时，采用基于距离的正负样本采样方法
		# 需要判断病例数据是否能够满足样本采样的需求
		assert self.num_positive_img+self.num_negative_img+2*self.minimum_negative_distance <= TotalNumImg, f'病例{patient_name}图像数量不足，无法满足采样需求'
		return {'NumSamples': self.num_negative_img+self.num_positive_img ,'NumLabel': TotalNumLabel}

	def GetLabelImgAndOriginalImgPath(self, patient_name:str, phase:str):
		assert phase in ['N','A','PV','V','D']
		if self.dataset[patient_name]['ImgLabelPairStatus'] == 'normal':
			# 若为normal模式，则其标注图像和原始图像的文件名不同
			label_path = self._GetPath(patient_name, 'img_in_label', phase)
			img_path = self._GetPath(patient_name, 'img', phase, self.dataset[patient_name][phase]['label_index'])
		else:
			raise Exception(f'Patient {patient_name} Phase {phase} is not in normal/label_only mode')
		return label_path, img_path

	def _SelectPatientAccodingToSplit(self, split:str):
		assert split in ['train', 'val', 'test']
		if hasattr(self, 'split'):
			warnings.warn(f"检测到对数据集split的重复定义: {self.split}->{split}")
		self.split = split
		# 按照split和self.split_ratio计算patient的姓名列表
		train_start_index = 0
		train_last_index = train_start_index + int(self.dataset['METADATA']['num_patient'] * self.split_ratio['train']) - 1
		val_start_index = train_last_index + 1
		val_last_index = val_start_index + int(self.dataset['METADATA']['num_patient'] * self.split_ratio['val']) - 1
		test_start_index = val_last_index + 1
		test_last_index = self.dataset['METADATA']['num_patient'] - 1
		split_index = {
			'train':	[train_start_index, train_last_index],
			'val':		[val_start_index,   val_last_index  ],
			'test':		[test_start_index,  test_last_index ],
		}
		all_patient_name = sorted(list(self.dataset.keys() - {'METADATA'}))
		self.activated_patient_name = all_patient_name[split_index[self.split][0]:split_index[self.split][1]+1]

	def setting(self, split:str, pretraining:bool, num_positive_img:int,
			 num_negative_img:int, minimum_negative_distance:int, ensambled_img_group:bool):
		'''
			split: 'train', 'val', 'test'，病例级划分
			pretraining: True/False，如果是预训练，则只会返回包含img图片项的字典，且会调用所有可用png图像；如果不是，则会去除一些没有标注的img
			num_positive_img: int，数据增强参数，指定中心标注周围多少个样本作为辅助正样本
			num_negative_img: int，数据增强参数，指定外围多少个样本作为负样本
			minimum_negative_distance: int，数据增强，指定负样本距离中心标注的最小距离（距离单位:样本）
			ensambled_img_group: bool，返回模式。如果True：返回单个病例的所有正负样本及中心标注；如果False：返回一个img、一个label，数据集长度较长。
		'''

		self.pretraining = pretraining
		self.num_positive_img = num_positive_img
		self.num_negative_img = num_negative_img
		self.minimum_negative_distance = minimum_negative_distance
		self.ensambled_img_group = ensambled_img_group
		
		# 非预训练情况需要进行样本分配参数检查
		if self.pretraining is False:
			if self.num_positive_img < 1:
				raise Exception('num_positive_img must be >= 1')
			if self.minimum_negative_distance < 1:
				raise Exception('minimum_negative_distance must be >= 1')
			# num_positive_img必须是奇数、num_negative_img必须是偶数，实现对称的数据增强
			if num_positive_img % 2 == 0:
				raise Exception('num_positive_img must be odd')
			if num_negative_img % 2 == 1:
				raise Exception('num_negative_img must be even')
		self._SelectPatientAccodingToSplit(split)	# 切分数据集
		self._InitGetItemIndex()	# 生成直接关联dataloader index的索引列表 self.IndexList

	def mmseg_load_data_list(self) -> list[dict]:
		# self.IndexList: [img_path, label_path, [patient, phase, index]]
		assert hasattr(self, 'IndexList'), 'IndexList not Initialized, call setting() first'
		data_list = []
		for sample in self.IndexList:
			info = {'img_path': sample[0], 
					'seg_map_path': sample[1], 
					'label_map': None, 
					'reduce_zero_label': False, 
					'seg_fields': []}
			data_list.append(info)
		
		# data_list: [{'img_path': str, 
		# 				'seg_map_path': str, 
		# 				'label_map': None, 
		# 				'reduce_zero_label':bool, 
		# 				'seg_fields': empty_list
		# 			}, ...]
		data_list = sorted(data_list, key=lambda x: x['img_path'])
		return data_list

	# 2024.3.14：标准分割任务数据集支持算法升级
	def MMSEG_Segmentation_PosNegEnhance(self, 
										split:str, num_positive_img:int, num_negative_img:int, 
										minimum_negative_distance:int):
		assert num_positive_img % 2 == 1, 'num_positive_img must be odd'
		assert num_negative_img % 2 == 0, 'num_negative_img must be even'
		assert minimum_negative_distance >= 1, 'minimum_negative_distance must be >= 1'

		self._SelectPatientAccodingToSplit(split)	# 切分数据集
		self.num_positive_img = num_positive_img
		self.num_negative_img = num_negative_img
		self.minimum_negative_distance = minimum_negative_distance
		datalist = []   # 索引缓冲区

		pbar = tqdm.tqdm(self.activated_patient_name, desc=f"[Dataset] {split}初始化", total=len(self.activated_patient_name))
		for patient_name in pbar:
			patient_data = self.dataset[patient_name]
			if patient_data['ImgLabelPairStatus'] != 'normal': 
				continue	# 跳过受限样本（只有标注或只有扫描图）
			for phase in ['N', 'A', 'PV', 'V', 'D']:
				if phase not in patient_data: continue
				if self.dataset[patient_name]['ImgLabelPairStatus'] == 'normal' or self.dataset[patient_name]['ImgLabelPairStatus'] == 'CTEnhanceScanPhaseLost':
					# 判断该扫描期是否有对应的标注
					if self.dataset[patient_name][phase]['LabeledImgFileName'] is None: continue
					# 生成正负样本的index索引范围
					PositiveImageIndexStart = self.dataset[patient_name][phase]['label_index'] - (self.num_positive_img-1)//2
					PositiveImageIndexList = list(range(PositiveImageIndexStart, PositiveImageIndexStart+self.num_positive_img))
					NegativeImageIndexRangeLeft = range(0, self.dataset[patient_name][phase]['label_index'] - self.minimum_negative_distance)
					NegativeImageIndexRangeRight = range(self.dataset[patient_name][phase]['label_index'] + self.minimum_negative_distance + 1, 
										  				 self.dataset[patient_name][phase]['NumImgFile'])
					NegativeImageIndexList = list(NegativeImageIndexRangeLeft)+list(NegativeImageIndexRangeRight)
					# 从NegativeImageIndexList中随机抽取num_negative_img个样本作为负样本
					np.random.shuffle(NegativeImageIndexList)	# 原地操作
					NegativeImageIndexList = NegativeImageIndexList[:self.num_negative_img]

					for ImgIndex in PositiveImageIndexList + NegativeImageIndexList:
						img_path = self._GetPath(patient_name, 'img', phase, ImgIndex)
						label_path = self._GetPath(patient_name, 'label', phase) \
										if ImgIndex in PositiveImageIndexList else None
						sample = {
							'img_path': img_path, 
							'seg_map_path': label_path, 
							'label_map': None, 
							'reduce_zero_label': False, 
							'seg_fields': [],
							'metadata': self.dataset[patient_name][phase]['metadata']
						}
						datalist.append(sample)
				
				# 由于没有原始扫描图片，无法进行基于距离的数据增强的病例
				elif self.dataset[patient_name]['ImgLabelPairStatus'] == 'label_only':
					img_path = self._GetPath(patient_name, 'img_in_label', phase)
					label_path = self._GetPath(patient_name, 'label', phase)
					sample = {
							'img_path': img_path, 
							'seg_map_path': label_path, 
							'label_map': None, 
							'reduce_zero_label': False, 
							'seg_fields': [],
							'metadata': self.dataset[patient_name][phase]['metadata']
						}
					datalist.append(sample)

				elif self.dataset[patient_name]['ImgLabelPairStatus'] == 'img_only':
					continue	# 只有扫描图像没有标注的病例只能用于无监督预训练
			
			pbar.update()
		pbar.close()
		return datalist

	# 自监督相对位置学习预训练任务
	# data_list: [{'img_path': str, 
	# 				'gt_label': None(mmpretrain无监督训练)
	# 			}, ...]
	# samples_per_scan: 从一次扫描序列中提取的样本数量
	def MMPRETASK_RelativeIndex_LoadDataList(
			self, split:str, gt_field:str, samples_per_scan:int=1, 
			minimum_negative_distance:int=50, distance_clip:int=500) -> List[Dict]:
		assert samples_per_scan >= 1, '至少从一次扫描序列中提取一个样本'
		assert gt_field in ['normal', 'abs', 'binary']  # abs:绝对值 index_binary:±1
		self.gt_field = gt_field
		self.minimum_negative_distance = minimum_negative_distance
		self.distance_clip = distance_clip

		self._SelectPatientAccodingToSplit(split)	# 切分数据集
		datalist = []   # 索引缓冲区

		for patient_name in self.activated_patient_name:
			patient_data = self.dataset[patient_name]
			if patient_data['ImgLabelPairStatus'] != 'normal': 
				continue	# 跳过受限样本（只有标注或只有扫描图）

			for phase in ['N', 'A', 'PV', 'V', 'D']:
				if phase not in patient_data: continue
				
				# 获取病灶中心位置
				center_slice_idx = patient_data[phase]['label_index']
				if center_slice_idx is None: continue	# 有的病人有多个扫描期，但不是所有的扫描器都有标注，此时ImgLabelPairStatus 依旧为normal
				
				# 获取该扫描期所有可用样本的序列号
				slice_idx_available = list(range(patient_data[phase]['NumImgFile']))
				
				# 随机采样
				np.random.shuffle(slice_idx_available)
				latter_slice_idx = samples_per_scan if samples_per_scan <= len(slice_idx_available) else len(slice_idx_available)
				rand_selected_slices_idxs = slice_idx_available[:latter_slice_idx]

				# 记录样本
				for rand_slice_idx in rand_selected_slices_idxs:
					# 计算归一化的样本与病灶相对位置
					if self.gt_field == 'normal':
						relative_index_residual = rand_slice_idx-center_slice_idx
					elif self.gt_field == 'abs':
						relative_index_residual = abs(rand_slice_idx-center_slice_idx)
					elif self.gt_field == 'binary':
						relative_index_residual = 1 if rand_slice_idx-center_slice_idx > 0 else -1 if rand_slice_idx!=center_slice_idx else 0
					if relative_index_residual >= self.distance_clip: relative_index_residual = self.distance_clip-1
					if relative_index_residual <= -self.distance_clip: relative_index_residual = -self.distance_clip+1
					# 获取样本路径
					img_path = self._GetPath(patient_name, 'img', phase, rand_slice_idx)
					# 记录
					datalist.append({'img_path': img_path, 'gt_label': relative_index_residual})
		
		return datalist

	def MMPRETASK_RelativeY_LoadDataList(
			self, split:str, gt_field:str, 
			samples_per_scan:int=1, 
			Y_filter:List[float]=[50,1000]) -> List[Dict]:
		assert samples_per_scan >= 1, '至少从一次扫描序列中提取一个样本'
		assert gt_field in ['normal', 'abs', 'binary']  # abs:绝对值 index_binary:±1
		self.gt_field = gt_field

		self._SelectPatientAccodingToSplit(split)	# 切分数据集
		datalist = []   # 索引缓冲区

		for patient_name in self.activated_patient_name:
			patient_data = self.dataset[patient_name]
			if patient_data['ImgLabelPairStatus'] != 'normal': 
				continue	# 跳过受限样本（只有标注或只有扫描图）

			for phase in ['N', 'A', 'PV', 'V', 'D']:
				scan_sample = []
				if phase not in patient_data: continue
				
				center_slice_Y = patient_data[phase]['label_Y']
				if center_slice_Y is None: continue	# 有的病人有多个扫描期，但不是所有的扫描器都有标注，此时ImgLabelPairStatus 依旧为normal
				index_Y_dict =   patient_data[phase]['index_Y_dict']
				series_Y_min =   patient_data[phase]['series_Y_range'][0] + 1

				# 获取可用idx列表
				slice_idx_available = list(range(patient_data[phase]['NumImgFile']))
				# 记录样本
				for idx in slice_idx_available:
					# 计算归一化的样本与病灶相对位置
					if self.gt_field == 'normal':
						relative_Y_residual = index_Y_dict[idx] - center_slice_Y
					elif self.gt_field == 'abs':
						relative_Y_residual = abs(index_Y_dict[idx] - center_slice_Y)
					elif self.gt_field == 'binary':
						relative_Y_residual = 1 if (index_Y_dict[idx]-center_slice_Y>0) else -1 if (index_Y_dict[idx]!=center_slice_Y) else 0
					# 判断relative_Y_residual是否介于Y_filter值域间
					if (relative_Y_residual >= Y_filter[0]) and (relative_Y_residual <= Y_filter[1]):
						img_path = self._GetPath(patient_name, 'img', phase, idx)
						scan_sample.append({'img_path': img_path, 'gt_label': int(relative_Y_residual+series_Y_min)})
				# 随机采样
				np.random.shuffle(scan_sample)
				datalist += scan_sample[:min(samples_per_scan, len(scan_sample))]
		
		return datalist

	def __len__(self):
		# 如果self.IndexList不存在，说明没有初始化GetItem的索引
		if not hasattr(self, 'IndexList'):
			raise Exception('IndexList not found, call InitGetItemIndex() first')
		return len(self.IndexList)

	def __getitem__(self, index):
		# self.IndexList: [img_path, label_path, [patient, phase, index]]

		if self.ensambled_img_group:
			raise NotImplementedError('ensambled_img_group function not implemented')
		
		META_DATA = self.IndexList[index][2]
		# 如果是预训练，则只返回img，不返回label
		if self.pretraining:
			img = np.load(self.IndexList[index][0])
			img = torch.from_numpy(img).squeeze()
			label = torch.from_numpy(self.EmptyLabel)
			return {'img': img, 'label': label, 'META_DATA': META_DATA}
		else:
			img = np.load(self.IndexList[index][0])
			label = np.load(self.IndexList[index][1]) if self.IndexList[index][1] is not None else self.EmptyLabel
			img = torch.from_numpy(img).squeeze()
			label = torch.from_numpy(label).squeeze().float()
			return {'img': img, 'label': label, 'META_DATA': META_DATA}



class MMPreSampleProvider(CQKGastricCancerCT):
	def MMPRE_RawList(self, split:str, samples_per_scan:int, *args, **kwargs) -> List[Dict]:
		assert samples_per_scan >= 1, '至少从一次扫描序列中提取一个样本'

		self._SelectPatientAccodingToSplit(split)	# 切分数据集
		datalist = []   # 索引缓冲区

		pbar = tqdm.tqdm(self.activated_patient_name, desc=f"正在索引{split}", total=len(self.activated_patient_name))
		for patient_name in pbar:
			patient_data = self.dataset[patient_name]
			if patient_data['ImgLabelPairStatus'] != 'normal': 
				continue	# 跳过受限样本（只有标注或只有扫描图）

			for phase in ['N', 'A', 'PV', 'V', 'D']:
				if phase not in patient_data: continue
				
				center_slice_Y = patient_data[phase]['label_Y']
				if center_slice_Y is None: continue	# 有的病人有多个扫描期，但不是所有的扫描器都有标注，此时ImgLabelPairStatus 依旧为normal

				# 获取可用idx列表
				slice_idx_available = list(range(patient_data[phase]['NumImgFile']))
				# 随机采样
				np.random.shuffle(slice_idx_available)
				latter_slice_idx = samples_per_scan if samples_per_scan <= len(slice_idx_available) else len(slice_idx_available)
				rand_selected_slices_idxs = slice_idx_available[:latter_slice_idx]

				# 记录样本
				for rand_slice_idx in rand_selected_slices_idxs:
					# 获取样本路径
					img_path = self._GetPath(patient_name, 'img', phase, rand_slice_idx)
					# 记录
					datalist.append({'img_path': img_path})
		
		return datalist






if __name__ == '__main__':
	root = r'D:/PostGraduate/DL/ClassicDataset/2023_Med_CQK'
	ckpt = r"D:/PostGraduate/DL/mgam_CT\data/2023_Med_CQK/V3_2024-03-15_15-15-59.pickle"
	dataset = CQKGastricCancerCT(root)

	for split in ['train']:
		file_list = dataset.MMSEG_Segmentation_PosNegEnhance(split=split, 
													   		 num_positive_img=3, 
															 num_negative_img=4, 
															 minimum_negative_distance=50)
		print(f'{split} file_list len: {len(file_list)}')

	import pdb; pdb.set_trace()






