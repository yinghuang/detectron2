## FCOS基于Detectron2的实现

**FCOS: Fully Convolutional One-Stage Object Detection** [paper](https://arxiv.org/abs/1904.01355), [official code](https://github.com/tianzhi0549/FCOS)  

代码参考[BorderDet](https://github.com/Megvii-BaseDetection/BorderDet/tree/master/playground/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x),
[BorderDet/cvpods/modeling/meta_arch/fcos.py](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/cvpods/modeling/meta_arch/fcos.py),
[预训练模型](https://drive.google.com/file/d/1hcDobxvqolMwqj20BEAPikSMcz4NYZRx/view)


### 在COCO 2017数据集上训练(带测试)demo

0.打开命令窗口, 激活conda环境(假设环境名为`detectron2_custom`), 切换至当前工程目录(注意不要跟`detectron2/detectron2/projects`搞混)
```
conda activate detectron2_custom
cd detectron2/projects/FCOS
```


1.确定确保xxx下数据集格式为:
```
xxx:
    coco:
        annotations:
            instances_train2017.json
            instances_val2017.json
        train2017:
            000000000xxx.jpg
            ...
        val2017:
            000000000xxx.jpg
            ...
    ...other datasets...
```
COCO数据集[主页](https://cocodataset.org/#download)  
COCO 2017 annotations可以从这里下载[annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)  
COCO 2017 images可以从这里下载[train2017.zip](http://images.cocodataset.org/zips/train2017.zip), [val2017](http://images.cocodataset.org/zips/val2017.zip)  


2.设置数据集根目录环境变量(detectron2在导入时会自动注册COCO这些预设的数据集, 而数据集的根目录是根据环境变量`DETECTRON2_DATASETS`来指定的. 参考`detectron2/detectron2/data/datasets/__init__.py`和`detectron2/detectron2/data/datasets/builtin.py`)
```
set DETECTRON2_DATASETS=xxx
```


3.下载[FCOS_imprv_R_50_FPN_1x.pth](https://drive.google.com/file/d/1hcDobxvqolMwqj20BEAPikSMcz4NYZRx/view)(页面顶部的预训练模型), 放在`detectron2/projects/FCOS/pretrained_weights`下


4.开始训练, 其他参数可以看`train_net.py`中`default_argument_parser`函数
```
python train_net.py ^
--config-file configs/fcos_R_50_FPN.yaml
```


5.开始训练后, 会在`cfg.OUTPUT_DIR`下输出一些文件  
`event....`: tfrecord训练日志文件  
`config.yaml`: 一份完整的config(注意里面的格式变了)  
`config_my.yaml`: 一份来自`--config-file`的拷贝文件  
`log.txt`: 控制台输出日志  
`metrics.json`: 训练输出日志  

或者以上所有命令集合在`train_fcos.bat`, 修改里面的Anaconda3路径后, 可以直接双击执行

### 图片推理demo

模型下载放置完毕后, 修改里面的Anaconda3路径, 直接双击执行`demo_inference_fcos.bat`

### 在自定义数据集上训练

1.写好yaml配置文件

2.通过`dataset/voc2coco.py`来制作coco标注文件(.json)

2.执行
```
python train_net.py ^
--config-file configs/fcos_R_50_FPN_develop.yaml ^
--ann_file_train xxx.json ^
--image_dir_train xxx ^
--num-gpus xxx
```

### Notes

1. 暂不支持cfg.MODEL.FCOS.NORM_SYNC=True, 需要判断当前是否处于多GPU训练
2. 如果要多GPU训练, cfg.MODEL.FCOS.IN_FEATURES 确保IN_FEATURES包括["p3", "p4", "p5", "p6", "p7"], 缺少一个就会报错 
"Expected to have finished reduction in the prior iteration before starting a new one"

### Detectron2 理解
1. `cfg.MIN_SIZE_TRAIN`和`cfg.MAX_SIZE_TRAIN`, 可以实现多尺度训练. 对batch中每张图片都进行keep ratio random resize, 
    但是最后给模型forward的batch中每个图片的shape是一样的, 具体在meta_arch的preprocess_image中, 
    使用batch中最大的长和宽定义了一个tensor, 每个图片分别放上去, size不够的就在右边和下边pad, 默认填充0像素值.







