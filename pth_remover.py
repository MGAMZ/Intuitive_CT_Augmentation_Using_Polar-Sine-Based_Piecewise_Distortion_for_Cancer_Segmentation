import argparse
import os

def remove_all_pth_files(root_path:str):
	count = 0
	for root, dirs, files in os.walk(root_path):
		for file in files:
			if file.endswith(".pth"):
				os.remove(os.path.join(root, file))
				print(f"Deleted: {os.path.join(root, file)}")
				count += 1
	print("\nDONE: remove {} pth files\n".format(count))


def args_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('root', type=str)
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = args_parser()
	exec_path = args.root

	# 确认“确认所有实验已经完成”，输入yes才继续运行
	print(f"WARNING: 即将递归遍历删除所有pth文件:{args.root}\n")
	print("确认所有实验已经完成？")
	confirm = input()
	if confirm != "yes":
		exit()
	else:
		# WARNING: 递归遍历删除所有pth文件
		remove_all_pth_files(exec_path)