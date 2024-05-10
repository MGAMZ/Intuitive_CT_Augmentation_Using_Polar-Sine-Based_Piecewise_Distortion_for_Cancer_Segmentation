import os
import re
import subprocess
import argparse
from copy import deepcopy


def ExistPthFile(work_dir_path) -> bool:
    if not os.path.exists(work_dir_path): 
        return False
    for file in os.listdir(work_dir_path):
        if file.endswith(".pth"):
            return True
    return False


def VersionToFullExpName(args):
    if args.exp_name[-1] == ".":
        raise AttributeError(f"目标实验名不得以“.”结尾：{args.exp_name}")
    exp_list = os.listdir(args.config_root)
    for exp in exp_list:
        if exp == args.exp_name:
            print(f"已找到实验：{args.exp_name} <-> {exp}")
            return exp
        elif exp.startswith(args.exp_name):
            pattern = r'\.[a-zA-Z]'    # 正则表达式找到第一次出现"."与字母连续出现的位置
            match = re.search(pattern, exp)
            if exp[:match.start()] == args.exp_name:
                print(f"已根据实验号找到实验：{args.exp_name} -> {exp}")
                return exp
            # if not "." in exp[len(args.exp_name)+1:]:       # 序列号后方不得再有“.”
            #     if not exp[len(args.exp_name)+1].isdigit(): # 序列号若接了数字，说明该目录实验是目标实验的子实验
                    # print(f"已根据实验号找到实验：{args.exp_name} -> {exp}")
                    # return exp
    raise RuntimeError(f"未找到与“ {args.exp_name} ”匹配的实验名")


def runner(args):
    print(f"{args.exp_name} 实验启动")
    
    for round in range(1, args.exp_trials+1):
        exp_round = os.path.join(args.work_dir_root, f"{args.exp_name}/round_{round}")

        for model in args.models:
            config_path = os.path.join(args.config_root, f"{args.exp_name}/{model}.py")
            work_dir_path = f"{exp_round}/{model}"
            # 设置终端标题
            if os.name == 'nt':
                os.system(f"title round {round} {model} - {args.exp_name} ")
            else:
                print(f"\n\n\n--------- title round {round} {model} - {args.exp_name} ---------\n\n\n")
            
            # pth文件只会在训练结束后保存一次，中途不保存pth
            if ExistPthFile(work_dir_path): 
                print(f"{args.exp_name} 第{round}次 {model} 实验已经完成, 跳过: {work_dir_path}")
                continue
            
            print(f"{args.exp_name} 第{round}次 {model} 实验开始")
            # 启动实验
            cmd = f"python tools/train.py {config_path} --work-dir {work_dir_path}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"{args.exp_name} 第{round}次 {model} 实验失败", e)
                exit(1)
            print(f"{args.exp_name} 第{round}次 {model} 实验完成")


def parase_args():
    parser = argparse.ArgumentParser(description="MMPRETRAIN")
    parser.add_argument("exp_name", type=str, nargs="+", help="实验名")
    parser.add_argument("--models", type=str, nargs="+", default=["ConvNext", "MAE", "Poolformer", "Resnet50", "Segformer", "SegNext", "SwinTransformerV2"])
    parser.add_argument("--exp_trials", type=int, default=1, help="每个实验重复次数")
    parser.add_argument("--work_dir_root", type=str, default="work_dirs", help="存储实验结果的根目录")
    parser.add_argument("--config_root", type=str, default="configs/mgam", help="存储配置文件的根目录")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    # 获取当前终端启动的路径
    current_path = os.getcwd()
    if os.path.basename(current_path)!="mmpretrain" or not os.path.exists("mmpretrain"):
        raise RuntimeError("当前终端路径异常，请确认在 mmsegmentation 目录下启动")
    # 获取参数
    args = parase_args()

    for exp in args.exp_name:
        sub_args = deepcopy(args)   # 隔离参数
        sub_args.exp_name = exp
        sub_args.exp_name = VersionToFullExpName(sub_args)  # 自动补全实验名
        runner(sub_args) # 启动实验
        print(f"{sub_args.exp_name} 实验结束")