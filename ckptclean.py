import os
import re
import argparse
from copy import deepcopy
from pprint import pprint


def VersionToFullExpName(args):
    if args.exp_name[-1] == ".":
        raise AttributeError(f"目标实验名不得以“.”结尾：{args.exp_name}")
    exp_list = os.listdir(args.root)
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
    raise RuntimeError(f"未找到与“ {args.exp_name} ”匹配的实验名")


def clean_pth(args):
    root = os.path.join(args.root, args.exp_name)
    print(f"Scanning directory: {root}\n")
    
    found_pth_paths = []
    for roots, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".pth"):
                found_pth_paths.append(os.path.join(roots, file))
    
    print(f"Found {len(found_pth_paths)} .pth files:\n")
    pprint(found_pth_paths)
    confirm = input(f'\nConfirm Clean All {len(found_pth_paths)} pth files from {args.exp_name}? (y/n)')
    
    if confirm.lower() == 'y':
        for pth_path in found_pth_paths:
            os.remove(pth_path)
        print(f"Cleaned {len(found_pth_paths)} .pth files.\n")
    elif confirm.lower() == 'n':
        print("No files were cleaned. Exit.\n")
    else:
        print("Invalid input. Exit.\n")


DEFAULT_CLEAN_PATH = './mm_work_dirs'

def parser_args():
    parser = argparse.ArgumentParser(description="Clean all .pth files in a directory.")
    parser.add_argument("exp_name", type=str, nargs="+", help="Specify Experiment Version to Be Cleaned")
    parser.add_argument("--root", type=str, default=DEFAULT_CLEAN_PATH, help="Root directory path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    for exp in args.exp_name:
        sub_args = deepcopy(args)
        sub_args.exp_name = exp
        exp_name = VersionToFullExpName(sub_args)
        sub_args.exp_name = exp_name
        clean_pth(sub_args)


