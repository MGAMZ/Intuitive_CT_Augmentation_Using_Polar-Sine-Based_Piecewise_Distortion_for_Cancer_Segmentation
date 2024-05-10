import os
import subprocess
import argparse


def parase_args():
	parser = argparse.ArgumentParser(description="Run experiment.")
	parser.add_argument("--exp-name", type=str)
	parser.add_argument("--models", type=str, nargs="+", default=["ConvNext", "MAE", "Poolformer", "Resnet50", "Segformer", "SegNext", "SwinTransformerV2"])
	args = parser.parse_args()
	return args


def ExistPthFile(work_dir_path) -> bool:
	if not os.path.exists(work_dir_path): 
		return False
	for file in os.listdir(work_dir_path):
		if file.endswith(".pth"):
			return True
	return False		


if __name__ == "__main__":
	# 获取当前终端启动的路径
	current_path = os.getcwd()
	assert os.path.basename(current_path)=="mmseg", "请在 mmsegmentation 目录下启动"

	args = parase_args()
	print("实验启动")

	for round in range(1, 4):
		print(f"第{round}次实验开始")
		exp_round = f"work_dirs/{args.exp_name}/round_{round}"

		for model in args.models:
			config_path = f"configs/mgam/{args.exp_name}/{model}.py"
			work_dir_path = f"{exp_round}/{model}"
			
			if ExistPthFile(work_dir_path): 
				print(f"{work_dir_path} 已经完成, 跳过")
				continue
			
			cmd = f"python {config_path} --work-dir {work_dir_path}"
			subprocess.run(cmd, shell=True, check=True)
			print(f"{model} 实验完成")

	print("实验结束")