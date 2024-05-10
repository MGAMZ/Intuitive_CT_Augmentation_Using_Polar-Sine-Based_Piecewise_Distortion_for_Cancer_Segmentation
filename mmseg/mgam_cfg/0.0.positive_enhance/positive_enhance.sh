set +e
echo "实验启动：正样本增强实验"

for round in 1 2 3 4 5
do

echo "第${round}次实验"
exp_round="work_dirs/PositiveEnhance_${round}"
python tools/train.py configs/mgam/1.positive_enhance/ConvNext.py --work-dir "${exp_round}/ConvNext"
python tools/train.py configs/mgam/1.positive_enhance/MAE.py --work-dir "${exp_round}/MAE"
python tools/train.py configs/mgam/1.positive_enhance/Poolformer.py --work-dir "${exp_round}/Poolformer"
python tools/train.py configs/mgam/1.positive_enhance/Resnet50.py --work-dir "${exp_round}/Resnet50"
python tools/train.py configs/mgam/1.positive_enhance/Segformer.py --work-dir "${exp_round}/Segformer"
python tools/train.py configs/mgam/1.positive_enhance/SegNext.py --work-dir "${exp_round}/SegNext"
python tools/train.py configs/mgam/1.positive_enhance/SwinTransformerV2.py --work-dir "${exp_round}/SwinTransformerV2"

done

echo "实验结束：正样本增强实验"