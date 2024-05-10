set -e

echo "实验启动"

exp_name="2.smooth_corner_stretch"

round=1
echo "第${round}次实验"
exp_round="work_dirs/${exp_name}/round_${round}"

# python tools/train.py configs/mgam/${exp_name}/ConvNext.py --work-dir "${exp_round}/ConvNext"
# python tools/train.py configs/mgam/${exp_name}/MAE.py --work-dir "${exp_round}/MAE"
python tools/train.py configs/mgam/${exp_name}/Poolformer.py --work-dir "${exp_round}/Poolformer"
python tools/train.py configs/mgam/${exp_name}/Resnet50.py --work-dir "${exp_round}/Resnet50"
python tools/train.py configs/mgam/${exp_name}/Segformer.py --work-dir "${exp_round}/Segformer"
python tools/train.py configs/mgam/${exp_name}/SegNext.py --work-dir "${exp_round}/SegNext"
python tools/train.py configs/mgam/${exp_name}/SwinTransformerV2.py --work-dir "${exp_round}/SwinTransformerV2"

round=2
echo "第${round}次实验"
exp_round="work_dirs/${exp_name}/round_${round}"

python tools/train.py configs/mgam/${exp_name}/ConvNext.py --work-dir "${exp_round}/ConvNext"
python tools/train.py configs/mgam/${exp_name}/MAE.py --work-dir "${exp_round}/MAE"
python tools/train.py configs/mgam/${exp_name}/Poolformer.py --work-dir "${exp_round}/Poolformer"
python tools/train.py configs/mgam/${exp_name}/Resnet50.py --work-dir "${exp_round}/Resnet50"
python tools/train.py configs/mgam/${exp_name}/Segformer.py --work-dir "${exp_round}/Segformer"
python tools/train.py configs/mgam/${exp_name}/SegNext.py --work-dir "${exp_round}/SegNext"
python tools/train.py configs/mgam/${exp_name}/SwinTransformerV2.py --work-dir "${exp_round}/SwinTransformerV2"

round=3
echo "第${round}次实验"
exp_round="work_dirs/${exp_name}/round_${round}"

python tools/train.py configs/mgam/${exp_name}/ConvNext.py --work-dir "${exp_round}/ConvNext"
python tools/train.py configs/mgam/${exp_name}/MAE.py --work-dir "${exp_round}/MAE"
python tools/train.py configs/mgam/${exp_name}/Poolformer.py --work-dir "${exp_round}/Poolformer"
python tools/train.py configs/mgam/${exp_name}/Resnet50.py --work-dir "${exp_round}/Resnet50"
python tools/train.py configs/mgam/${exp_name}/Segformer.py --work-dir "${exp_round}/Segformer"
python tools/train.py configs/mgam/${exp_name}/SegNext.py --work-dir "${exp_round}/SegNext"
python tools/train.py configs/mgam/${exp_name}/SwinTransformerV2.py --work-dir "${exp_round}/SwinTransformerV2"

echo "实验结束"