## FCOS基于Detectron2的实现

[FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355), [official code](https://github.com/tianzhi0549/FCOS)  

代码参考[BorderDet](https://github.com/Megvii-BaseDetection/BorderDet/tree/master/playground/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x),
[fcos.py](https://github.com/Megvii-BaseDetection/BorderDet/blob/master/cvpods/modeling/meta_arch/fcos.py)
[预训练模型](https://drive.google.com/file/d/1hcDobxvqolMwqj20BEAPikSMcz4NYZRx/view)


### 在COCO 2017数据集上训练(带测试)demo

0. 打开命令窗口, 激活conda环境(假设环境名为`detectron2_custom`), 切换至当前工程目录(注意不要跟`detectron2/detectron2/projects`搞混)
```
conda activate detectron2_custom
cd detectron2/projects/FCOS
```


1. 确定确保xxx下数据集格式为:
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


2. 设置数据集根目录环境变量(detectron2在导入时会自动注册COCO这些预设的数据集, 而数据集的根目录是根据环境变量`DETECTRON2_DATASETS`来指定的. 参考`detectron2/detectron2/data/datasets/__init__.py`和`detectron2/detectron2/data/datasets/builtin.py`)
```
set DETECTRON2_DATASETS=xxx
```


3. 下载[FCOS_imprv_R_50_FPN_1x.pth](https://drive.google.com/file/d/1hcDobxvqolMwqj20BEAPikSMcz4NYZRx/view)(页面顶部的预训练模型), 放在`detectron2/projects/FCOS/pretrained_weights`下


4. 开始训练, 其他参数可以看`train_net.py`中`default_argument_parser`函数
```
python train_net.py ^
--config-file configs/fcos_R_50_FPN.yaml
```


5. 开始训练后, 会在`cfg.OUTPUT_DIR`下输出一些文件  
`event....`: tfrecord训练日志文件  
`config.yaml`: 一份完整的config(注意里面的格式变了)  
`config_my.yaml`: 一份来自`--config-file`的拷贝文件  
`log.txt`: 控制台输出日志  
`metrics.json`: 训练输出日志  


或者以上所有命令集合在`train_fcos.bat`, 可以直接双击执行

### 图片推理demo

与训练模型下载后, 直接双击执行`demo_inference_fcos.bat`

