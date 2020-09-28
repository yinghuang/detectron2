@echo off
set PATH=%PATH%;D:\code\Anaconda3;D:\code\Anaconda3\Scripts;D:\code\Anaconda3\Library\bin
set DETECTRON2_DATASETS=D:\dataset
call activate.bat detectron2_custom
set PYTHONPATH=%PYTHONPATH%;D:\code\Anaconda3\envs\detectron2_custom;D:\code\Anaconda3\envs\detectron2_custom\Scripts;D:\code\Anaconda3\envs\detectron2_custom\DLLs;D:\code\Anaconda3\envs\detectron2_custom\Lib;
cmd.exe /k python train_net.py ^
--config-file configs/centernet_R_50_FPN_develop.yaml ^
--dataset_dir D:\code\detectron2_custom\detectron2\projects\FCOS\data\tmp ^
--dataset_settxt paths.txt ^
--class_names hb ^
--dataset_register_type absolute ^
--num-gpus 1