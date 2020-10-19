@echo off
set PATH=%PATH%;D:\code\Anaconda3;D:\code\Anaconda3\Scripts;D:\code\Anaconda3\Library\bin
call activate.bat detectron2_custom
set PYTHONPATH=%PYTHONPATH%;D:\code\Anaconda3\envs\detectron2_custom;D:\code\Anaconda3\envs\detectron2_custom\Scripts;D:\code\Anaconda3\envs\detectron2_custom\DLLs;D:\code\Anaconda3\envs\detectron2_custom\Lib;
cmd.exe /k python demo.py --config-file configs/fcos_R_50_FPN.yaml ^
--input ../../demo/coco_val2017_000000002153.jpg ../../demo/coco_val2017_000000002685.jpg ^
--opts MODEL.WEIGHTS pretrained_weights/FCOS_imprv_R_50_FPN_1x.pth