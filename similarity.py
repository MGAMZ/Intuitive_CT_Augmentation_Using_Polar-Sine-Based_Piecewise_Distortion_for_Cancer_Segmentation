import cv2, pdb, os, datetime, pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, List, Union, Dict
from tqdm import tqdm


from mmseg.CQK_Med import Distortion, CQK_2023_Med





class SimilarityMetric:
	def __init__(self, mask:np.ndarray):

		self.mask = mask


	def __call__(self, img1, img2):
		raise NotImplementedError



class SIFT(SimilarityMetric):
	def __init__(self, kdtree:int=1, num_trees:int=5, checks:int=10, **kwargs) -> None:
		super().__init__(**kwargs)
		self.sift = cv2.SIFT_create()
		index_params = dict(algorithm=kdtree, trees=num_trees)
		search_params = dict(checks=checks)
		self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

	def __call__(self, img1, img2):
		kp1, des1 = self.sift.detectAndCompute(img1, self.mask)
		kp2, des2 = self.sift.detectAndCompute(img2, self.mask)
		return self.matcher.knnMatch(des1,des2,k=2)



class ORB(SimilarityMetric):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.orb = cv2.ORB_create()
		self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

	def __call__(self, img1, img2):
		kp1 = self.orb.detect(img1, self.mask)
		kp1, des1 = self.orb.compute(img1, kp1)
		kp2 = self.orb.detect(img2, self.mask)
		kp2, des2 = self.orb.compute(img2, kp2)
		return self.matcher.knnMatch(des1, des2, k=2)



class SimilarityStatistics:
	SUPPORTED_SIMILARITY_METRICS = ['SIFT', 'ORB']

	def __init__(self, log_path:str, distorter:Distortion=None,
				 img_shape=(512,512), mask_ratio:float=0.6, ):
		self.img_shape = img_shape
		self.distorter = distorter
		self.mask = self._get_mask(img_shape, mask_ratio)
		self.log_path = log_path

	@staticmethod
	def _read_npy(path:str) -> np.ndarray:
		img:np.ndarray = np.load(path)
		img = np.clip(img,0,4095).astype(np.uint8)
		return img
	
	@staticmethod
	def _get_mask(shape, ratio) -> np.ndarray:
		if isinstance(shape, Iterable):
			assert shape[0] == shape[1]
			shape=shape[0]
		max_radius = shape // 2
		radius = int(max_radius * ratio)  # 半径为边长的四分之一
		y, x = np.ogrid[-max_radius:max_radius, -max_radius:max_radius]
		mask_circle = x**2 + y**2 <= radius**2
		
		return mask_circle.astype(np.uint8)

	@staticmethod
	def _get_npy_from_path(root:str) -> List[str]:
		assert os.path.exists(root)
		npys = []
		pbar = tqdm(desc='Find npy Files')
		for roots, dirs, files in os.walk(root):
			for file in files:
				if file.endswith('.npy'):
					npys.append(os.path.join(roots, file))
					pbar.update()
		pbar.close()
		return npys


	def _set_matcher(self, matchers:Union[str, Iterable[str]]):  # 暂时不支持动态切换
		if not isinstance(matchers, Iterable):
			matchers = [matchers]
		self.similariry_metric_name = matchers
		self.matcher = []
		
		for matcher in matchers:
			if matcher == 'SIFT':
				self.matcher.append(SIFT(mask=self.mask, kdtree=1, num_trees=5, checks=10)) 
			elif matcher == 'ORB':
				self.matcher.append(ORB(mask=self.mask))
			else:
				raise NotImplementedError


	def AnalyzeOneImg(self, img1, img2):
		result = []
		for match in self.matcher:
			matches = match(img1, img2)
			good = []
			for m,n in matches:
				if m.distance < 0.5*n.distance:
					good.append(m)
			result.append(len(good))
		
		return result


	def AnalyzeImgs(self, ndarray_list:List[np.ndarray]) -> np.ndarray:
		num_mean_matches = []
		img1s = ndarray_list
		img2s = [img.astype(np.uint8) for img in self.distorter.multiprocess_distort(img1s)]
		for img1, img2 in tqdm(zip(img1s, img2s), desc='Matching', total=len(img1s)):
			result:List[int, int] = self.AnalyzeOneImg(img1, img2)
			num_mean_matches.append(result)
		
		matches = np.array(num_mean_matches).mean(axis=0)
		return matches

	# 自动化消融实验实现
	def Ablation(self, npys:Union[str, Iterable],
				 similarity_metrics:Iterable[str],
				 A_s=[0.5,1,2,3,4,5,6,7,8], 
				 F_s=[0.5,1,2,3,4,5,6,7,8], 
				 ) -> List[Dict]:
		if hasattr(self,'distorter'):
			print('Distortor Already Initialized, Will be Overwritten...')
		
		# 设定所需要的匹配算法，可以是多个
		self._set_matcher(similarity_metrics)
		
		# 初始化日志
		time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		log_folder = os.path.join(self.log_path, f'SimilarityAblation_{time}')
		if not os.path.exists(log_folder):
			os.mkdir(log_folder)
		log_txt = open(os.path.join(log_folder, f'meta.txt'), 'w',)
		
		SimilarityConfigurations = pprint.pformat(self.__dict__, indent=4)
		print(SimilarityConfigurations)
		log_txt.write('Similarity Configurations: \n\n' + SimilarityConfigurations + '\n\n\n\n')
		log_txt.close()

		# 随机选样本，加载
		if isinstance(npys, str):
			npys = self._get_npy_from_path(npys)
		elif isinstance(npys, Iterable):
			pass
		else:
			raise TypeError('Must be str or Iterable')
		npys = np.random.choice(npys, 100, replace=False)
		ori_imgs = [self._read_npy(npy) for npy in npys]

		# 准备dataframe
		dfs = [pd.DataFrame(index=A_s,columns=F_s) for _ in range(len(similarity_metrics))]

		# 执行
		pbar = tqdm(desc='Running Ablations', total=len(A_s)*len(F_s))
		for A in A_s:
			for F in F_s:
				pbar.set_description(f"Running Ablation | A={A}, F={F}")
				# 扭曲处理
				self.distorter = Distortion(global_rotate=0, amplitude=A, frequency=F,
											grid_dense=32, in_array_shape=(512,512), 
											refresh_interval=10000, const=True)
				# 相似性计算
				num_mean_matches = self.AnalyzeImgs(ori_imgs)
				# 记录
				for i, metric_name in enumerate(similarity_metrics):
					dfs[i].loc[A,F] = num_mean_matches[i]
				pbar.update()
		pbar.close()

		# 根据result，存储为xlsx文件
		log_xlsx = os.path.join(log_folder, f'AF_Distribution.xlsx')
		writer = pd.ExcelWriter(log_xlsx)
		for df, metric_name in zip(dfs, similarity_metrics):
			df.to_excel(writer, sheet_name=metric_name)
			print(f'Log Saved to xlsx: {log_xlsx} | sheet: {metric_name} ')
		writer.close()


	def Visualize(self, img1, img2, kp1, kp2, matches):
		MIN_MATCH_COUNT = 10
		if len(matches)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			matchesMask = mask.ravel().tolist()
			h,w = self.img_shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,M)
			img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		else:
			print( "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
			matchesMask = None

		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
							singlePointColor = None,
							matchesMask = matchesMask, # draw only inliers
							flags = 2)
		vis = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
		# plt.imshow(vis, 'gray'), plt.show()
		return vis


	def __exit__(self, exc_type, exc_value, traceback):
		if hasattr(self, 'l'):
			self.l.flush()
			self.l.close()





if __name__ == "__main__":
	database_args={
		'root': './2023_Med_CQK', 
		'metadata_ckpt': './2023_Med_CQK/V3_2024-03-16_14-59-01.pickle',
		'split': 'train',
		'num_positive_img': 1,
		'num_negative_img': 50,
		'minimum_negative_distance': 1,
	}
	npy_list = CQK_2023_Med(database_args)._DATABASE.MMPRETASK_RelativeY_LoadDataList(
				'train', 'normal', 1, Y_filter=[-5,5])
	npy_list = [d['img_path'] for d in npy_list]

	log_path = './result/Similarity/'
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	Summarizer = SimilarityStatistics(log_path)
	Summarizer.Ablation(npys=npy_list, 
						similarity_metrics=['SIFT', 'ORB'])




