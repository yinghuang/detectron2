@echo off
set PATH=%PATH%;D:\code\Anaconda3;D:\code\Anaconda3\Scripts;D:\code\Anaconda3\Library\bin
call activate.bat rknn
cmd.exe /k  python voc2coco.py ^
--ann_dir D:\dataset\pospal_hb\labeled\Annotations ^
--ann_ids D:\dataset\pospal_hb\labeled\sets\all_24965_id.txt ^
--labels D:\dataset\pospal_hb\labeled\class_names.txt ^
--ext xml ^
--output D:\dataset\pospal_hb\labeled\coco_annotations\all_24965_id.json ^