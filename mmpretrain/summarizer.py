import os, argparse, re
from typing import Dict, List, Tuple, AnyStr, Iterable
from multiprocessing import Pool

import colorama
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 存储每个模型每个实验每个loss的reduction后的值
PER_EXP_LOSS = Iterable[Tuple[AnyStr, Dict[str, float]]]
# 存储每个模型每个loss在多次实验中的mean和std值
AVG_LOSS = Dict[AnyStr, Dict[str, Tuple[float, float]]]


# 从一个tsb日志中取出一个系列的所有值，并可reduction
# 由于使用了多进程，传参时打包为了tuple
def get_tsb_serial_avg(params:tuple):
    try:
        file, reduction, loss_names = params
        assert callable(reduction)
        assert isinstance(file, str), f"file must be str, but got {type(file)}"
        assert isinstance(loss_names, Iterable), f"loss_names must be list, but got {type(loss_names)}"
        assert isinstance(loss_names[0], str), f"loss_names must be list of str, but got {loss_names[0]} in {type(loss_names)}"
        assert "events.out.tfevents" in file, f"file must be a tsb file, but got {file}"

        event = EventAccumulator(file)
        event.Reload()
        loss_values = {}

        for series in loss_names:
            loss_value = [scalar.value for scalar in event.Scalars(series)]
            loss_values[series] = reduction(loss_value)
    
    except Exception as e:
        print(f"Failed at {params}, info: {e}")
        return e
        
    return loss_values


# 找到某项实验中某个模型的多次实验的所有tsb日志文件
def get_tensorboard_filelist(workdir_root:str, model_list:list|tuple, exp:str) -> Dict[str, List[str]]:
    workdir_root =os.path.join(workdir_root, exp) # 同一参数的多次实验被放在了一个文件夹内
    all_model_exp_tsb_files = {}    # dict{'model_name': [tsb1,tsb2,...]}

    # 分别为每个模型查找属于它们的tsb文件
    for model_name in model_list:
        single_model_tsb_files = []
        # 查找一个模型每次实验的tsb文件
        for exp in os.listdir(workdir_root):    # 获取实验轮次文件夹列表
            tensorboard_files = []
            # 在一次实验中找到指定的模型实验
            models = [model for model in os.listdir(os.path.join(workdir_root, exp)) 
                      if model_name in model]
            for model in models:
                for root, dirs, files in os.walk(os.path.join(workdir_root, exp, model)):
                    for file in files:
                        if file.startswith("events.out.tfevents"):
                            tensorboard_files.append(os.path.join(root, file))
            # 正常情况下，一次实验一个模型只对应一个tensorboard日志
            # mmseg每次启动脚本都会生成一个实验目录，包含一个tsb文件。
            # 在mmseg中，设定为只在训练结束时保存一次pth。
            # 在run的脚本中，若检测到该次实验文件夹下已经存在pth文件，则判定为已经完成，将会跳过脚本
            # 根据启动时间取最后生成的tsb日志，因为其代表该次实验的最后一次执行
            if len(tensorboard_files)>1:
                time = 0    # 临时变量：用于查询哪个是最大（最晚）的时间
                for tsb_file_path in tensorboard_files:
                    # 取父文件夹的父文件夹名，即启动时间。日志方法由mmengine定义
                    dir_name_of_exp_time = os.path.basename(os.path.dirname(os.path.dirname(tsb_file_path)))
                    # 文件夹名形如：20240101_010203，需要去除中间的“_”
                    exp_time = dir_name_of_exp_time.replace("_", "")
                    assert len(exp_time) == 14
                    exp_time = int(exp_time)
                    if exp_time > time:
                        time = exp_time
                        latest_exp_tsb_file_path = tsb_file_path
                single_model_tsb_files.append(latest_exp_tsb_file_path)
            
            elif len(tensorboard_files)==1:
                single_model_tsb_files.append(tensorboard_files[0])

        if len(single_model_tsb_files) == 0:
            raise FileNotFoundError(f"model {model_name} has no tsb files")
        all_model_exp_tsb_files[model_name] = single_model_tsb_files
        print(f"model {model_name} has {len(single_model_tsb_files)} tsb files")

    return all_model_exp_tsb_files


# 多进程调用
def multi_process(func, param_list:list, n_jobs:int):
    with Pool(n_jobs) as pool:
        with tqdm(total=len(param_list), desc='MultiProcess Summarizing') as pbar:
            values = []
            for i, value in enumerate(pool.imap(func, param_list)):
                if isinstance(value, Exception):
                    print(colorama.Fore.RED + f"Failed at {param_list[i]}, info: {value}" + colorama.Style.RESET_ALL)
                    raise value
                values.append(value)    # dict{'Loss_Name':float,...}
                pbar.update()
    return values   # list[dict{'Loss_Name':float,...}, dict, ...]


def result_collect(results:PER_EXP_LOSS) -> AVG_LOSS:
    # model_name_list: list[str]
    # results: list[dict{'Loss_Name':float,...}, dict, ...]
    # 分类合并所有实验结果
    model_losses = {}
    for model_name, model_result in results:
        if model_name not in model_losses.keys():
            model_losses[model_name] = {}
        for loss_name, loss_value in model_result.items():
            if loss_name not in model_losses[model_name].keys():
                model_losses[model_name][loss_name] = []
            model_losses[model_name][loss_name].append(loss_value)
    # 分组计算所有实验的mean和std
    for model_name, model_loss in model_losses.items():
        for loss_name, loss_value_list in model_loss.items():
            mean_value = float(np.mean(loss_value_list))
            std_value = float(np.std(loss_value_list))
            model_losses[model_name][loss_name] = (mean_value, std_value)    # tuple(mean, std)

    return model_losses


def tabulate_print(result: AVG_LOSS) -> None:
    # 随机取出一个model，用于获取所有loss_name
    model_name = list(result.keys())[0]
    loss_name_list = list(result[model_name].keys())
    headers = ['model_name', *loss_name_list]
    table = []

    for model_name, loss_dict in result.items():
        row = [model_name]+ [f"{loss_dict[loss_name][0]:.4f}±{loss_dict[loss_name][1]:.4f}"
                             for loss_name in loss_name_list]
        table.append(row)
    print(tabulate(table, headers, tablefmt="grid"))


def csv_save(result, csv_save_path:str):
    assert csv_save_path.endswith(".csv"), "csv_save_path must be a csv file"
    
    if not os.path.exists(os.path.dirname(csv_save_path)):
        os.makedirs(os.path.dirname(csv_save_path))

    df = pd.DataFrame(result).T
    df_mean_std = df.copy()
    df_mean = df.copy()
    df_std = df.copy()
    
    # Convert each item from (mean, std) to mean±std
    for col in df_mean_std.columns:
        df_mean_std[col] = df_mean_std[col].apply(lambda x: f"{x[0]:.4f}±{x[1]:.4f}")
    # Convert each item from (mean, std) to mean
    for col in df_mean.columns:
        df_mean[col] = df_mean[col].apply(lambda x: f'{x[0]:.4f}')
    # Convert each item from (mean, std) to std
    for col in df_std.columns:
        df_std[col] = df_std[col].apply(lambda x: f'{x[1]:.4f}')

    # 将三个df纵向concat，并保存为csv
    df_all = pd.concat([df_mean_std, df_mean, df_std], axis=1)
    df_all.to_csv(csv_save_path, encoding="utf-8-sig")


# 根据exp版本号查找完整exp名
def VersionToFullExpName(args):
    if args.exp_name[-1] == ".":
        raise AttributeError(f"目标实验名不得以“.”结尾：{args.exp_name}")
    exp_list = os.listdir(args.workdir_root)
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


def get_reduction_func(args):
    assert args.reduction in ('mean', 'std', 'min', 'max')

    if args.reduction == 'mean':
        return np.mean
    elif args.reduction == 'std':
        return np.std
    elif args.reduction == 'min':
        return min
    elif args.reduction == 'max':
        return max

def parse_args():
    parser = argparse.ArgumentParser(description='Summarize the result of mmsegmentation')
    parser.add_argument('exp_name', help='The experiment name', type=str)
    parser.add_argument('--workdir_root', help='The root path of workdir', type=str, default='./work_dirs')
    parser.add_argument('--csv_save_path', help='The path to save csv file', type=str, default='../result/mmpretrain')
    parser.add_argument('--model_list', help='The list of model name', nargs="+", type=str, default=['ConvNext', 'MAE', 'Poolformer', 'Resnet50', 'Segformer', 'SegNext', 'SwinTransformerV2'])
    parser.add_argument('--loss_name', help='The list of loss name', nargs="+", type=str, default='Dist/RelPos_L1')
    parser.add_argument('--reduction', type=str, default='min')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.exp = VersionToFullExpName(args)   # 自动补全文件名
    if isinstance(args.loss_name, str):
        args.loss_name = [args.loss_name]
    # 收集tsb信息
    tsb_list_all_model = get_tensorboard_filelist(args.workdir_root, args.model_list, exp=args.exp) # tsb_list: dict{'model_name': [tsb1,tsb2,...]}
    exec_param = []
    model_name_list = []
    for model_name, tsb_list in tsb_list_all_model.items():
        for tsb_file in tsb_list:
            exec_param.append((tsb_file, min, args.loss_name))  # IoU Dice Recall Precision
            model_name_list.append(model_name)
    
    # 多进程解析tsb
    values = multi_process(get_tsb_serial_avg, exec_param, 20)
    metric:PER_EXP_LOSS = zip(model_name_list, values)
    # 单进程计算度量
    result = result_collect(metric)

    # 打印和保存
    tabulate_print(result)
    csv_save(result, os.path.join(args.csv_save_path, f"{args.exp}.csv"))




