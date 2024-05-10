import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS as MMSEG_MODELS
from mmseg.utils import ConfigType
from mmseg.models.segmentors import EncoderDecoder
from mmpretrain.registry import MODELS as MMPRE_MODELS
from mmpretrain.models.backbones import MobileNetV3
from mmengine.model import BaseModule, ModuleList


# 集成中心矫正前置环节得EncoderDecoder模型, 将扫描区域移动至感受野中心, 提高模型利用率
@MMSEG_MODELS.register_module()
class CenterCorrectionModel(EncoderDecoder):
    def __init__(self, 
                 centercorrectionmodel:ConfigType, 
                 pad_value:int|float=0, 
                 *args, **kwargs):
        super(CenterCorrectionModel, self).__init__(*args, **kwargs)
        
        self.centercorrectionmodel = MMSEG_MODELS.build(centercorrectionmodel)  # 中心矫正模型（前置阶段）
        self.pad_value = pad_value
        # 禁用梯度
        self.centercorrectionmodel.eval()
        self.centercorrectionmodel.requires_grad_(False)

    # 根据中心索引对数据进行移位，空位补0
    def centercorrection(self, in_array:torch.Tensor, center_index:torch.Tensor):
        # array:(B, C, H, W)
        # center_index:(B, 2)  2指的是二维图像索引值
        B, H, W, C = in_array.shape
        array = in_array.clone()
        center_index = torch.clip(center_index, min=0, max=max(in_array.shape)-1)
        
        for b in range(B):
            shift = (int(H//2-center_index[b,0]), int(W//2-center_index[b,1]))
            # roll循环移位
            array[b] = torch.roll(array[b], shifts=shift, dims=(0, 1))
            # 移位反方向区域置默认值
            if shift[0] > 0:
                array[b, :shift[0], :, :] = self.pad_value
            elif shift[0] < 0:
                array[b, shift[0]:, :, :] = self.pad_value
            if shift[1] > 0:
                array[b, :, :shift[1], :] = self.pad_value
            elif shift[1] < 0:
                array[b, :, shift[1]:, :] = self.pad_value
        return array

    def extract_feat(self, inputs: torch.Tensor):	# x:(B, C, H, W)
        # centercorrection
        with torch.no_grad():
            center_index = self.centercorrectionmodel(inputs)	# 在backbone之前先定位内容中心点 center_index:(B, 2)
            x = self.centercorrection(inputs, center_index)	# 按照内容中心点对图像移位
        # backbone
        x = super().extract_feat(x)
        
        return x


class CCHead(BaseModule):
    def __init__(self, in_feature_map_size:tuple, mid_channel:int=2, num_cnn_downsampler:int=4,
                kernel_size=5, stride=1, padding=2, **kwargs):
        assert len(in_feature_map_size) == 3, "in_feature_map_size shape should be (C, H, W), but got {}".format(in_feature_map_size)
        assert in_feature_map_size[1]==in_feature_map_size[2], "in_feature_map_size shape should be a square, but got {}".format(in_feature_map_size)
        super().__init__(**kwargs)

        # 下采样块
        cnn_downsampler_channel_list = [in_feature_map_size[0]//2**i for i in range(num_cnn_downsampler+1)]
        self.cnn_downsampler = torch.nn.ModuleList([
            torch.nn.Conv2d(cnn_downsampler_channel_list[i], cnn_downsampler_channel_list[i+1], kernel_size=kernel_size, stride=stride, padding=padding)
            for i in range(num_cnn_downsampler)
            ] + [
            torch.nn.Conv2d(cnn_downsampler_channel_list[-1], mid_channel, kernel_size=1, stride=1, padding=0)
            ])
        
        # 全连接输出
        size = in_feature_map_size[1]
        for _ in range(num_cnn_downsampler):
            size = math.floor((size - kernel_size + 2*padding) / stride) + 1
        size = (mid_channel, size, size)
        self.fc = torch.nn.Linear(size[0]*size[1]*size[2], 2)
    
    def forward(self, x):
        assert len(x.shape) == 4, "input shape should be (B, C, H, W), but got {}".format(x.shape)
        
        for downsampler in self.cnn_downsampler:
            x = downsampler(x)
        x = self.fc(x.flatten(start_dim=1))
        
        return x


# CC: CenterCorrection
@MMPRE_MODELS.register_module()
class MobileNetV3_CC(MobileNetV3):
    def __init__(self, *args, **kwargs):
        super(MobileNetV3_CC, self).__init__(arch='small', in_channels=1, *args, **kwargs)
        shape = (576, 8, 8)
        self.head = CCHead(shape, mid_channel=16, num_cnn_downsampler=2, 
                kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = super().forward(x)[0]
        x = self.head(x)
        return x


if __name__ == '__main__':
    from mmpretrain.models.backbones import ConvNeXt

    backbone_model = ConvNeXt(arch='base', in_channels=1, out_indices=(0,1,2,3), use_grn=True, gap_before_final_norm=False)
    centercorrection_model = MobileNetV3_CC()
    model = CenterCorrectionModel(backbone_model, centercorrection_model)
    
    batch_size = 2
    shape = (1, 256, 256)	# (C, H, W)
    data = torch.randn(batch_size, *shape)
    output = model(data)
    import pdb;pdb.set_trace()

