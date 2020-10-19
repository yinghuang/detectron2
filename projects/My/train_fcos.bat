@echo off
set PATH=%PATH%;D:\code\Anaconda3;D:\code\Anaconda3\Scripts;D:\code\Anaconda3\Library\bin
set DETECTRON2_DATASETS=D:\dataset
call activate.bat detectron2_custom
set PYTHONPATH=%PYTHONPATH%;D:\code\Anaconda3\envs\detectron2_custom;D:\code\Anaconda3\envs\detectron2_custom\Scripts;D:\code\Anaconda3\envs\detectron2_custom\DLLs;D:\code\Anaconda3\envs\detectron2_custom\Lib;
cmd.exe /k python train_net.py ^
--config-file configs/fcos_R_50_FPN.yaml