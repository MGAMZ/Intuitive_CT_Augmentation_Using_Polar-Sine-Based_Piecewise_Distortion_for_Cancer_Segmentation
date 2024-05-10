# 出错也不终止
set +e
echo "实验启动：baseline"

python tools/train.py configs/mgam/0.baseline/BASELINE_ConvNext.py
python tools/train.py configs/mgam/0.baseline/BASELINE_MAE.py
python tools/train.py configs/mgam/0.baseline/BASELINE_Poolformer.py
python tools/train.py configs/mgam/0.baseline/BASELINE_Resnet50.py
python tools/train.py configs/mgam/0.baseline/BASELINE_Segformer.py
python tools/train.py configs/mgam/0.baseline/BASELINE_SegNext.py
python tools/train.py configs/mgam/0.baseline/BASELINE_SwinTransformerV2.py

echo "实验结束：baseline"