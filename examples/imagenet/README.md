# Example on ImageNet

## 数据预处理

把coco数据转换成FFReocrd：

```
python imagenet2ffrecord.py
```
生成的FFRecord文件保存在`/private_dataset/ImageNet/train.ffr`和`/private_dataset/ImageNet/val.ffr`

## 训练

本地训练：
```
python train.py
```

