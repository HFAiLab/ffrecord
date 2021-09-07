# SSD300

## 数据预处理

1. 把coco数据链接到`data/coco`文件夹：
    ```
    ln -s /path/to/coco data/coco
    ```
    `data/coco`应该有`train2017`和`val2017`两个文件夹

2. 把coco数据转换成FFReocrd：
    ```
    python coco2ffrecord.py
    ```
    生成的FFRecord文件保存在`data/coco/train2017.ffr`和`data/coco/val2017.ffr`

## 训练

本地训练：
```
python train.py
```
