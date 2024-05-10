import os
import numpy as np
from pprint import pprint
from summarizer import get_tsb_serial_avg

LOSS_NAME = ['mIoU', 'mDice', 'mRecall', 'mPrecision']

if __name__ == '__main__':
	root = 'D:\PostGraduate\DL\mgam_CT\mmseg\work_dirs\8.2.0.R600_Poolformer_Boost'
	tsb_files = []
	for roots, dirs, files in os.walk(root):
		for file in files:
			if file.startswith('events.out.tfevents'):
				tsb_files.append((os.path.join(roots, file), max, LOSS_NAME))
	
	results = []
	for tsb in tsb_files:
		results.append(get_tsb_serial_avg(tsb))
	
	collect = [[],[],[],[]]
	for i, loss_name in enumerate(LOSS_NAME):
		for result in results:
			collect[i].append(result[loss_name])
	for i in range(len(collect)):
		print(np.mean(collect[i]), np.std(collect[i]))

