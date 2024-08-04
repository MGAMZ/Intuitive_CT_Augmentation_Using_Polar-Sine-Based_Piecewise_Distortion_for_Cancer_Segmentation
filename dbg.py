import types

import torch
# Torch Profiler
from torch.profiler import profile, record_function, ProfilerActivity
from mmpretrain.models.backbones import SwinTransformerV2

class head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(start_dim=1)
        )
        self.linear1 = torch.nn.Linear(1024, 1)
        self.linear2 = torch.nn.Linear(512, 1)
        self.linear3 = torch.nn.Linear(256, 1)
        self.linear4 = torch.nn.Linear(128, 1)
        self.out = torch.nn.Linear(4, 1)
    
    def forward(self, backbone_list:list):
        # backbone 倒置
        backbone_list = backbone_list[::-1]
        hid = []
        for i, x in enumerate(backbone_list):
            temp = self.shared(x)
            temp = getattr(self, f"linear{i+1}")(temp)
            hid.append(temp)
        
        hid = torch.cat(hid, dim=1)
        return self.out(hid)
        



if __name__ == "__main__":
    # 启用tensorcore
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    import torch.cuda.profiler as profiler
    import pyprof
    pyprof.init()

    batch_size = 16
    model = SwinTransformerV2(
        arch='base',
        img_size=256,
        in_channels=1,
        out_indices=(0,1,2,3)
    ).cuda()
    head_model = head().cuda()
    optimizer = torch.optim.Adam(
        [
            {'params':model.parameters()},
            {'params':head_model.parameters()}
        ],
        lr=1e-3
    )
    scaler = torch.cuda.amp.GradScaler()
    data_in = torch.randn(1024, 1, 256, 256, dtype=torch.float, device=torch.device('cuda'))

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(r'D:\PostGraduate\DL\mgam_CT\dbg'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        ) as p:
            for iter in range(32):
                x = data_in[iter*batch_size:(iter+1)*batch_size].cuda()
                with torch.cuda.amp.autocast():
                    hid = model(x)
                    y = head_model(hid)
                    loss = torch.nn.functional.mse_loss(y, torch.rand_like(y))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                p.step()
